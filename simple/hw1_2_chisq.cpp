#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;
using Rng = std::mt19937_64;

const double INF = std::numeric_limits<double>::infinity();
const double PI = 3.14159265358979323846;

// =====================================================================
// [1] SETTINGS
// =====================================================================
const int R = 10000;
const std::vector<int> N_LIST = {2, 3, 4, 6, 11, 31, 101};   // df = n - 1 = 1, 2, 3, 5, 10, 30, 100
const unsigned SEED = 20261018;

const double POP_MU = 0, POP_SD = 2;   // every population is scaled to this mean and SD
const double T_NU = 10;                // t population df (> 4 so that Var(S^2) is finite)
const double ALPHA = 0.05;             // central 95% band of chi-square for the coverage check

// =====================================================================
// helpers
// =====================================================================
template <typename... Args>
std::string fmt(const char* f, Args... args) {
    char buf[512];
    std::snprintf(buf, sizeof buf, f, args...);
    return buf;
}

double normalPdf(double x, double mu, double var) {
    double z = (x - mu) / std::sqrt(var);
    return std::exp(-0.5 * z * z) / std::sqrt(2 * PI * var);
}

double chisqPdf(double x, double df) {
    if (x < 0) return 0.0;
    double k = df / 2;
    if (x == 0) return k < 1 ? INF : (k == 1 ? 0.5 : 0.0);
    return std::exp((k - 1) * std::log(x) - x / 2 - std::lgamma(k) - k * std::log(2.0));
}

// Regularized lower incomplete gamma P(a, x): series for x < a + 1, continued fraction otherwise.
double gammaP(double a, double x) {
    if (x <= 0) return 0.0;
    double lnPre = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1) {
        double ap = a, del = 1 / a, sum = del;
        for (int i = 0; i < 1000 && std::fabs(del) > std::fabs(sum) * 1e-15; ++i) {
            ap += 1;
            del *= x / ap;
            sum += del;
        }
        return sum * std::exp(lnPre);
    }
    double tiny = 1e-300, b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        double an = -i * (i - a);
        b += 2;
        d = an * d + b; if (std::fabs(d) < tiny) d = tiny;
        c = b + an / c; if (std::fabs(c) < tiny) c = tiny;
        d = 1 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1) < 1e-15) break;
    }
    return 1 - std::exp(lnPre) * h;
}

double chisqQuantile(double p, double df) {
    double lo = 0, hi = df + 50 * std::sqrt(2 * df) + 50;
    for (int i = 0; i < 200; ++i) {
        double mid = (lo + hi) / 2;
        (gammaP(df / 2, mid / 2) < p ? lo : hi) = mid;
    }
    return (lo + hi) / 2;
}

// =====================================================================
// [2] DISTRIBUTIONS - all populations share mean POP_MU and variance POP_SD^2
// =====================================================================
struct Dist {
    std::string key, label;
    double mu, var;
    double kurt;                         // excess kurtosis; drives Var((n-1)S^2/sigma^2)
    double lo, hi;                       // support
    std::function<double(Rng&)> draw;    // one random draw from the population
    std::function<double(double)> f;     // pdf
};

std::vector<Dist> makeDists() {
    std::vector<Dist> v;
    const double mu = POP_MU, var = POP_SD * POP_SD;
    {
        std::normal_distribution<double> d(POP_MU, POP_SD);
        v.push_back({"normal", fmt("Normal(μ = %g, σ = %g)", POP_MU, POP_SD),
                     mu, var, 0.0, -INF, INF,
                     [d](Rng& g) mutable { return d(g); },
                     [=](double x) { return normalPdf(x, mu, var); }});
    }
    {
        // X = mu + c * T, T ~ t(nu); Var(T) = nu / (nu - 2), so c rescales to Var(X) = sigma^2
        std::student_t_distribution<double> d(T_NU);
        const double nu = T_NU, c = POP_SD / std::sqrt(nu / (nu - 2));
        const double lnK = std::lgamma((nu + 1) / 2) - std::lgamma(nu / 2) - 0.5 * std::log(nu * PI);
        v.push_back({"t", fmt("scaled t(ν = %g), μ = %g, σ = %g", nu, POP_MU, POP_SD),
                     mu, var, 6 / (nu - 4), -INF, INF,
                     [d, c, mu](Rng& g) mutable { return mu + c * d(g); },
                     [=](double x) {
                         double t = (x - mu) / c;
                         return std::exp(lnK - (nu + 1) / 2 * std::log(1 + t * t / nu)) / c;
                     }});
    }
    {
        // Uniform(mu - a, mu + a) with a = sigma * sqrt(3) has variance sigma^2
        const double a = POP_SD * std::sqrt(3.0), lo = mu - a, hi = mu + a;
        std::uniform_real_distribution<double> d(lo, hi);
        v.push_back({"uniform", fmt("Uniform(%.3f, %.3f), μ = %g, σ = %g", lo, hi, POP_MU, POP_SD),
                     mu, var, -1.2, lo, hi,
                     [d](Rng& g) mutable { return d(g); },
                     [=](double x) { return (x >= lo && x <= hi) ? 1 / (hi - lo) : 0.0; }});
    }
    return v;
}

// =====================================================================
// [3] SAMPLING - RNG (mt19937_64, fixed seed per distribution and n) + S^2 loop
// =====================================================================
struct Sim { std::vector<double> q, s2; };   // q = (n-1) S^2 / sigma^2

Sim simulate(const Dist& d, int n, unsigned distIdx) {
    std::seed_seq ss{SEED, distIdx, unsigned(n)};
    Rng g(ss);
    Sim out{std::vector<double>(R), std::vector<double>(R)};
    std::vector<double> x(n);
    for (int r = 0; r < R; ++r) {
        double m = 0;
        for (int i = 0; i < n; ++i) { x[i] = d.draw(g); m += x[i]; }
        m /= n;
        double ss2 = 0;
        for (double v : x) ss2 += (v - m) * (v - m);
        out.s2[r] = ss2 / (n - 1);
        out.q[r] = ss2 / d.var;
    }
    return out;
}

struct Stats { double mean, var, skew; };

Stats describe(const std::vector<double>& x) {
    double m = 0, m2 = 0, m3 = 0;
    for (double v : x) m += v;
    m /= x.size();
    for (double v : x) { double e = v - m; m2 += e * e; m3 += e * e * e; }
    m2 /= x.size(); m3 /= x.size();
    return {m, m2 * x.size() / (x.size() - 1.0), m3 / std::pow(m2, 1.5)};
}

// =====================================================================
// histogram
// =====================================================================
struct Hist {
    std::vector<double> left, right, dens;
    std::vector<int> count;
    double width = 0, plotLo = 0, plotHi = 0;
};

// Bins are anchored at 0 (the chi-square support bound); range = 0.1% - 99.9% quantiles.
Hist makeHist(const std::vector<double>& x) {
    std::vector<double> s(x);
    std::sort(s.begin(), s.end());
    double qlo = s[size_t(0.001 * (R - 1))];
    double qhi = s[size_t(std::ceil(0.999 * (R - 1)))];

    Hist h;
    double w = (qhi - qlo) / 40;
    h.width = w;
    long kmin = long(std::floor(qlo / w));
    long kmax = long(std::ceil(qhi / w));
    if (kmax <= kmin) kmax = kmin + 1;
    h.count.assign(kmax - kmin, 0);
    for (double v : x) {
        long k = long(std::floor(v / w));
        if (k >= kmin && k < kmax) h.count[k - kmin]++;
    }
    for (long k = kmin; k < kmax; ++k) {
        h.left.push_back(k * w);
        h.right.push_back((k + 1) * w);
    }
    for (int c : h.count) h.dens.push_back(c / (double(R) * w));
    double a = h.left.front(), b = h.right.back(), pad = 0.05 * (b - a);
    h.plotLo = a - pad;
    h.plotHi = b + pad;
    return h;
}

// =====================================================================
// SVG drawing
// =====================================================================
const double PW = 400, PH = 340;
const double HEADER = 100;
const double ML = 60, MR = 18, MT = 66, MB = 46;

const char* C_BAR = "#7fb0dc";
const char* C_CHISQ = "#d1495b";
const char* C_POP = "#1f7a6d";
const char* C_REF = "#888888";
const char* C_TEXT = "#222222";
const char* C_SUB = "#555555";
const char* C_GRID = "#e4e4e4";
const char* FONT = "Segoe UI, Helvetica, Arial, sans-serif";

double niceStep(double raw) {
    double mag = std::pow(10, std::floor(std::log10(raw)));
    double r = raw / mag;
    return (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
}

std::string tickLabel(double v, double step) {
    int dec = std::max(0, -int(std::floor(std::log10(step) + 1e-9)));
    if (std::fabs(v) < step * 1e-6) v = 0;
    return fmt("%.*f", dec, v);
}

using Curve = std::vector<std::pair<double, double>>;

struct Panel {
    std::string title, info, info2, xlabel, ylabel;
    double xlo, xhi, ymax;
    const Hist* hist = nullptr;
    Curve chisq;   // chi-square(n-1) pdf
    Curve pdf;     // population pdf
    Curve ref;     // N(mu, sigma^2) reference (dashed)
};

void polyline(std::ostringstream& o, const Curve& c, const std::function<double(double)>& sx,
              const std::function<double(double)>& sy, const char* color, const char* extra) {
    std::ostringstream pts;
    for (auto& [x, y] : c) pts << fmt("%.2f,%.2f ", sx(x), sy(y));
    o << "<polyline points=\"" << pts.str() << "\" fill=\"none\" stroke=\"" << color << "\" " << extra << "/>\n";
}

void drawPanel(std::ostringstream& o, const Panel& p, double px, double py, int id) {
    double x0 = px + ML, x1 = px + PW - MR, y0 = py + PH - MB, y1 = py + MT;
    std::function<double(double)> sx = [&](double x) { return x0 + (x - p.xlo) / (p.xhi - p.xlo) * (x1 - x0); };
    std::function<double(double)> sy = [&](double y) { return y0 - std::min(y, p.ymax) / p.ymax * (y0 - y1); };

    o << fmt("<clipPath id=\"c%d\"><rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/></clipPath>\n",
             id, x0, y1, x1 - x0, y0 - y1);

    double ys = niceStep(p.ymax / 5);
    for (double y = 0; y <= p.ymax + 1e-12; y += ys) {
        o << fmt("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\"/>\n", x0, sy(y), x1, sy(y), C_GRID);
        o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"11\" text-anchor=\"end\" fill=\"%s\">%s</text>\n",
                 x0 - 6, sy(y) + 4, C_SUB, tickLabel(y, ys).c_str());
    }
    double xs = niceStep((p.xhi - p.xlo) / 6);
    for (double x = std::ceil(p.xlo / xs) * xs; x <= p.xhi + 1e-12; x += xs) {
        o << fmt("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\"/>\n", sx(x), y0, sx(x), y0 + 5, C_SUB);
        o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"11\" text-anchor=\"middle\" fill=\"%s\">%s</text>\n",
                 sx(x), y0 + 18, C_SUB, tickLabel(x, xs).c_str());
    }

    o << fmt("<g clip-path=\"url(#c%d)\">\n", id);
    if (p.hist) {
        const Hist& h = *p.hist;
        double barPx = h.width / (p.xhi - p.xlo) * (x1 - x0);
        std::string stroke = barPx > 4 ? " stroke=\"#ffffff\" stroke-width=\"0.6\"" : "";
        for (size_t i = 0; i < h.dens.size(); ++i) {
            if (h.count[i] == 0) continue;
            o << fmt("<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"%s\"%s/>\n",
                     sx(h.left[i]), sy(h.dens[i]), sx(h.right[i]) - sx(h.left[i]), y0 - sy(h.dens[i]),
                     C_BAR, stroke.c_str());
        }
    }
    if (!p.pdf.empty()) {
        std::ostringstream pts;
        for (auto& [x, y] : p.pdf) pts << fmt("%.2f,%.2f ", sx(x), sy(y));
        o << "<polygon points=\"" << fmt("%.2f,%.2f ", sx(p.pdf.front().first), y0) << pts.str()
          << fmt("%.2f,%.2f", sx(p.pdf.back().first), y0) << "\" fill=\"" << C_POP << "\" fill-opacity=\"0.18\"/>\n";
        o << "<polyline points=\"" << pts.str() << "\" fill=\"none\" stroke=\"" << C_POP << "\" stroke-width=\"2.2\"/>\n";
    }
    if (!p.ref.empty()) polyline(o, p.ref, sx, sy, C_REF, "stroke-width=\"1.6\" stroke-dasharray=\"6 4\"");
    if (!p.chisq.empty()) polyline(o, p.chisq, sx, sy, C_CHISQ, "stroke-width=\"2\"");
    o << "</g>\n";

    o << fmt("<polyline points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"none\" stroke=\"%s\"/>\n",
             x0, y1, x0, y0, x1, y0, C_SUB);
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"15\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
             x0, py + 22, C_TEXT, p.title.c_str());
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12\" fill=\"%s\">%s</text>\n", x0, py + 40, C_SUB, p.info.c_str());
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12\" fill=\"%s\">%s</text>\n", x0, py + 56, C_SUB, p.info2.c_str());
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12\" text-anchor=\"middle\" fill=\"%s\">%s</text>\n",
             (x0 + x1) / 2, y0 + 36, C_TEXT, p.xlabel.c_str());
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12\" text-anchor=\"middle\" fill=\"%s\" "
             "transform=\"rotate(-90 %.1f %.1f)\">%s</text>\n",
             px + 16, (y0 + y1) / 2, C_TEXT, px + 16, (y0 + y1) / 2, p.ylabel.c_str());
}

Curve sampleCurve(const std::function<double(double)>& f, double lo, double hi) {
    Curve c;
    for (int i = 0; i <= 400; ++i) {
        double x = lo + (hi - lo) * i / 400;
        c.push_back({x, f(x)});
    }
    return c;
}

// Extra points at the support bounds keep vertical edges (e.g. Uniform) sharp.
Curve pdfCurve(const Dist& d, double lo, double hi) {
    std::vector<double> xs;
    for (int i = 0; i <= 600; ++i) xs.push_back(lo + (hi - lo) * i / 600);
    for (double b : {d.lo, d.hi})
        if (std::isfinite(b) && b > lo && b < hi) {
            double eps = (hi - lo) * 1e-6;
            xs.push_back(b - eps); xs.push_back(b); xs.push_back(b + eps);
        }
    std::sort(xs.begin(), xs.end());
    Curve c;
    for (double x : xs) {
        double y = d.f(x);
        bool inside = (x >= d.lo && x <= d.hi);
        if (x == d.lo && std::isfinite(d.lo)) y = std::max(y, d.f(x + (hi - lo) * 1e-9));
        if (x == d.hi && std::isfinite(d.hi)) y = std::max(y, d.f(x - (hi - lo) * 1e-9));
        c.push_back({x, inside ? y : 0.0});
    }
    return c;
}

std::string svgHeader(double W, double H) {
    return fmt("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\" "
               "font-family=\"%s\">\n<rect width=\"100%%\" height=\"100%%\" fill=\"#ffffff\"/>\n",
               W, H, W, H, FONT);
}

void legendItem(std::ostringstream& o, double& x, double y, const std::string& swatch, const std::string& text) {
    o << swatch;
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12.5\" fill=\"%s\">%s</text>\n", x + 26, y + 4, C_TEXT, text.c_str());
    x += 34 + 6.6 * text.size();
}

std::string lineSwatch(double x, double y, const char* color, const char* extra) {
    return fmt("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"2.2\" %s/>\n",
               x, y, x + 20, y, color, extra);
}

// Use the wide-char API on Windows so a non-ASCII path is not mangled.
fs::path exeDir(char** argv) {
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 4];
    DWORD len = GetModuleFileNameW(nullptr, buf, DWORD(std::size(buf)));
    if (len > 0) return fs::path(std::wstring(buf, len)).parent_path();
#endif
    return fs::absolute(fs::path(argv[0])).parent_path();
}

// =====================================================================
// main
// =====================================================================
int main(int, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    fs::path outDir = exeDir(argv) / "output_1-2";
    fs::create_directories(outDir / "svg");
    fs::create_directories(outDir / "csv");

    const auto dists = makeDists();
    const std::string Q = "(n−1)S²/σ²";

    std::ofstream summary(outDir / "csv" / "summary.csv");
    summary << "distribution,n,df,mean_sim,mean_theory,var_sim,var_chisq,var_theory_pop,skew_sim,skew_chisq,"
               "coverage95_sim,mean_S2_sim,sigma2,var_S2_sim\n";

    std::printf("SQC HW1 #1-2  Sampling distribution of S²   R = %d, seed = %u\n", R, SEED);
    std::printf("Theory (normal population): (n−1)S²/σ² ~ χ²(ν), ν = n−1:  mean ν, var 2ν, skew √(8/ν)\n");
    std::printf("Any population with excess kurtosis κ:  var = 2ν + κ·ν²/n\n");

    for (unsigned di = 0; di < dists.size(); ++di) {
        const Dist& d = dists[di];
        std::vector<Sim> sims;
        std::vector<Hist> hists;
        std::vector<Panel> panels(1 + N_LIST.size());

        for (int n : N_LIST) {
            sims.push_back(simulate(d, n, di));
            hists.push_back(makeHist(sims.back().q));
        }

        std::printf("\n%s   σ² = %g, excess kurtosis κ = %g\n", d.label.c_str(), d.var, d.kurt);
        // header padded by hand: printf counts bytes, not display width, for ν / ² / √
        std::printf("      n    ν |    mean      ν |      var       2ν  2ν+κν²/n |   skew   √(8/ν) | in 95%% band\n");
        std::printf("  -----------+----------------+-----------------------------+-----------------+------------\n");

        // panels 2-8: (n-1)S^2/sigma^2 for each n
        std::ofstream hcsv(outDir / "csv" / ("hist_" + d.key + ".csv"));
        hcsv << "n,df,bin_left,bin_right,bin_mid,count,density,chisq_pdf\n";
        for (size_t j = 0; j < N_LIST.size(); ++j) {
            int n = N_LIST[j];
            double df = n - 1;
            const Hist& h = hists[j];
            Stats st = describe(sims[j].q);
            Stats s2 = describe(sims[j].s2);
            double tvar = 2 * df, tskew = std::sqrt(8 / df), popVar = 2 * df + d.kurt * df * df / n;
            double qL = chisqQuantile(ALPHA / 2, df), qU = chisqQuantile(1 - ALPHA / 2, df);
            int inside = 0;
            for (double v : sims[j].q) inside += (v >= qL && v <= qU);
            double cover = 100.0 * inside / R;

            Panel& p = panels[j + 1];
            p.title = fmt("n = %d   (ν = %d)", n, int(df));
            p.info = fmt("mean %.4g (χ² %g)  ·  var %.4g (χ² %g)", st.mean, df, st.var, tvar);
            p.info2 = fmt("skew %.3f (χ² %.3f)  ·  in 95%% band %.1f%%", st.skew, tskew, cover);
            p.xlabel = Q;
            p.ylabel = "density";
            p.xlo = h.plotLo; p.xhi = h.plotHi;
            p.hist = &h;
            p.chisq = sampleCurve([df](double x) { return chisqPdf(x, df); },
                                  std::max(p.xlo, 1e-9), p.xhi);
            // y range from the histogram and the curve at bin centres (chi-square(1) is infinite at 0)
            double ymax = 0;
            for (size_t i = 0; i < h.dens.size(); ++i)
                ymax = std::max({ymax, h.dens[i], chisqPdf((h.left[i] + h.right[i]) / 2, df)});
            p.ymax = ymax * 1.1;

            for (size_t i = 0; i < h.dens.size(); ++i) {
                double mid = (h.left[i] + h.right[i]) / 2;
                hcsv << n << ',' << df << ',' << h.left[i] << ',' << h.right[i] << ',' << mid << ',' << h.count[i]
                     << ',' << h.dens[i] << ',' << chisqPdf(mid, df) << '\n';
            }
            summary << d.key << ',' << n << ',' << df << ',' << st.mean << ',' << df << ',' << st.var << ','
                    << tvar << ',' << popVar << ',' << st.skew << ',' << tskew << ',' << cover << ','
                    << s2.mean << ',' << d.var << ',' << s2.var << '\n';
            std::printf("  %5d %4d | %7.3f %6d | %8.3f %8d %9.3f | %6.3f %8.3f | %9.2f%%\n",
                        n, int(df), st.mean, int(df), st.var, int(tvar), popVar, st.skew, tskew, cover);
        }

        // panel 1: population pdf on a common range so the three populations compare directly
        Panel& pop = panels[0];
        pop.title = "Population: " + d.key;
        pop.info = fmt("μ = %g  ·  σ² = %g  ·  skew 0", d.mu, d.var);
        pop.info2 = fmt("excess kurtosis κ = %g", d.kurt);
        pop.xlabel = "x";
        pop.ylabel = "f(x)";
        pop.xlo = d.mu - 4 * POP_SD; pop.xhi = d.mu + 4 * POP_SD;
        pop.pdf = pdfCurve(d, pop.xlo, pop.xhi);
        if (d.key != "normal")
            pop.ref = sampleCurve([&](double x) { return normalPdf(x, d.mu, d.var); }, pop.xlo, pop.xhi);
        std::ofstream pcsv(outDir / "csv" / ("population_" + d.key + ".csv"));
        pcsv << "x,pdf\n";
        double ymax = normalPdf(d.mu, d.mu, d.var);
        for (auto& [x, y] : pop.pdf) {
            pcsv << x << ',' << y << '\n';
            ymax = std::max(ymax, y);
        }
        pop.ymax = ymax * 1.15;

        std::ofstream rcsv(outDir / "csv" / ("raw_" + d.key + ".csv"));
        rcsv << "rep";
        for (int n : N_LIST) rcsv << ",n=" << n;
        rcsv << '\n';
        for (int r = 0; r < R; ++r) {
            rcsv << r + 1;
            for (size_t j = 0; j < N_LIST.size(); ++j) rcsv << ',' << sims[j].q[r];
            rcsv << '\n';
        }

        double W = 4 * PW + 20, H = HEADER + 2 * PH + 16;
        std::ostringstream o;
        o << svgHeader(W, H);
        o << fmt("<text x=\"30\" y=\"36\" font-size=\"21\" font-weight=\"700\" fill=\"%s\">"
                 "Sampling distribution of %s — %s</text>\n",
                 C_TEXT, Q.c_str(), d.label.c_str());
        o << fmt("<text x=\"30\" y=\"59\" font-size=\"12.5\" fill=\"%s\">R = %d replications per n, seed = %u. "
                 "Values in brackets are χ²(ν) theory; band = central 95%% of χ²(ν). "
                 "Each panel uses its own axis range.</text>\n",
                 C_SUB, R, SEED);
        double lx = 30, ly = 82;
        legendItem(o, lx, ly, fmt("<rect x=\"%.1f\" y=\"%.1f\" width=\"20\" height=\"11\" fill=\"%s\"/>\n", lx, ly - 5, C_BAR),
                   "simulated " + Q + " (histogram, density)");
        legendItem(o, lx, ly, lineSwatch(lx, ly, C_CHISQ, ""), "χ²(n−1) pdf");
        legendItem(o, lx, ly, lineSwatch(lx, ly, C_POP, ""), "population pdf");
        if (d.key != "normal")
            legendItem(o, lx, ly, lineSwatch(lx, ly, C_REF, "stroke-dasharray=\"6 4\""), "reference N(μ, σ²)");
        for (size_t i = 0; i < panels.size(); ++i) {
            double px = 10 + (i % 4) * PW, py = HEADER + (i / 4) * PH;
            drawPanel(o, panels[i], px, py, int(i));
        }
        o << "</svg>\n";
        std::ofstream(outDir / "svg" / (d.key + ".svg")) << o.str();
    }

    std::cout << "\nOutput written to " << (exeDir(argv).filename() / "output_1-2").u8string() << '\n';

#ifdef _WIN32
    // Launched by double-click: this process owns the console window, so keep it open.
    DWORD pids[2];
    if (GetConsoleProcessList(pids, 2) == 1) {
        std::cout << "\nPress Enter to close...";
        std::cin.get();
    }
#endif
    return 0;
}
