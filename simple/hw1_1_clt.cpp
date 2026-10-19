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
#define NOMINMAX   
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
const std::vector<int> N_LIST = {1, 2, 10, 30, 100};
const unsigned SEED = 20261018;

const double NORM_MU = 0, NORM_SD = 1;
const int BIN_M = 10;      const double BIN_P = 0.2;   // m = trials (n is the sample size)
const double POIS_LAMBDA = 1;
const double UNIF_A = 0, UNIF_B = 1;
const double GAMMA_ALPHA = 2, GAMMA_BETA = 1;          // shape alpha, scale beta (mean = alpha*beta)

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

// =====================================================================
// [2] DISTRIBUTIONS - add / change populations here
// =====================================================================
struct Dist {
    std::string key, label;
    bool discrete;
    double mu, var;
    double lo, hi;                       // support
    std::function<double(Rng&)> draw;    // one random draw from the population
    std::function<double(double)> f;     // pmf (discrete) or pdf (continuous)
};

std::vector<Dist> makeDists() {
    std::vector<Dist> v;
    {
        std::normal_distribution<double> d(NORM_MU, NORM_SD);
        const double mu = NORM_MU, var = NORM_SD * NORM_SD;
        v.push_back({"normal", fmt("Normal(\u03BC = %g, \u03C3 = %g)", NORM_MU, NORM_SD), false,
                     mu, var, -INF, INF,
                     [d](Rng& g) mutable { return d(g); },
                     [=](double x) { return normalPdf(x, mu, var); }});
    }
    {
        std::binomial_distribution<int> d(BIN_M, BIN_P);
        const int m = BIN_M;
        const double p = BIN_P, q = 1 - p;
        v.push_back({"binomial", fmt("Binomial(m = %d, p = %g)", m, p), true,
                     m * p, m * p * q, 0, double(m),
                     [d](Rng& g) mutable { return double(d(g)); },
                     [=](double x) {
                         int k = int(std::lround(x));
                         if (k < 0 || k > m) return 0.0;
                         return std::exp(std::lgamma(m + 1.0) - std::lgamma(k + 1.0) - std::lgamma(m - k + 1.0)
                                         + k * std::log(p) + (m - k) * std::log(q));
                     }});
    }
    {
        std::poisson_distribution<int> d(POIS_LAMBDA);
        const double lam = POIS_LAMBDA;
        v.push_back({"poisson", fmt("Poisson(\u03BB = %g)", lam), true,
                     lam, lam, 0, INF,
                     [d](Rng& g) mutable { return double(d(g)); },
                     [=](double x) {
                         int k = int(std::lround(x));
                         if (k < 0) return 0.0;
                         return std::exp(k * std::log(lam) - lam - std::lgamma(k + 1.0));
                     }});
    }
    {
        std::uniform_real_distribution<double> d(UNIF_A, UNIF_B);
        const double a = UNIF_A, b = UNIF_B;
        v.push_back({"uniform", fmt("Uniform(a = %g, b = %g)", a, b), false,
                     (a + b) / 2, (b - a) * (b - a) / 12, a, b,
                     [d](Rng& g) mutable { return d(g); },
                     [=](double x) { return (x >= a && x <= b) ? 1 / (b - a) : 0.0; }});
    }
    {
        std::gamma_distribution<double> d(GAMMA_ALPHA, GAMMA_BETA);   // (shape, scale)
        const double a = GAMMA_ALPHA, b = GAMMA_BETA;
        v.push_back({"gamma", fmt("Gamma(\u03B1 = %g, \u03B2 = %g)", a, b), false,
                     a * b, a * b * b, 0, INF,
                     [d](Rng& g) mutable { return d(g); },
                     [=](double x) {
                         if (x < 0) return 0.0;
                         if (x == 0) return a < 1 ? INF : (a == 1 ? 1 / b : 0.0);
                         return std::exp((a - 1) * std::log(x) - x / b - std::lgamma(a) - a * std::log(b));
                     }});
    }
    return v;
}

// =====================================================================
// [3] SAMPLING - RNG (mt19937_64, fixed seed per distribution and n) + x-bar loop
// =====================================================================
std::vector<double> simulateMeans(const Dist& d, int n, unsigned distIdx) {
    std::seed_seq ss{SEED, distIdx, unsigned(n)};
    Rng g(ss);
    std::vector<double> xbar(R);
    for (int r = 0; r < R; ++r) {
        double s = 0;
        for (int i = 0; i < n; ++i) s += d.draw(g);
        xbar[r] = s / n;
    }
    return xbar;
}

struct Stats { double mean, var; };

Stats describe(const std::vector<double>& x) {
    double m = 0, ss = 0;
    for (double v : x) m += v;
    m /= x.size();
    for (double v : x) ss += (v - m) * (v - m);
    return {m, ss / (x.size() - 1.0)};
}

// =====================================================================
// histogram
// =====================================================================
struct Hist {
    std::vector<double> left, right, dens;
    std::vector<int> count;
    double width = 0, plotLo = 0, plotHi = 0;
};

Hist makeHist(const Dist& d, int n, const std::vector<double>& x) {
    std::vector<double> s(x);
    std::sort(s.begin(), s.end());
    double qlo = s[size_t(0.001 * (R - 1))];
    double qhi = s[size_t(std::ceil(0.999 * (R - 1)))];

    Hist h;
    if (d.discrete) {
        h.width = 1.0 / n;
        long kmin = std::lround(qlo * n), kmax = std::lround(qhi * n);
        h.count.assign(kmax - kmin + 1, 0);
        for (double v : x) {
            long k = std::lround(v * n);
            if (k >= kmin && k <= kmax) h.count[k - kmin]++;
        }
        for (long k = kmin; k <= kmax; ++k) {
            h.left.push_back((k - 0.5) * h.width);
            h.right.push_back((k + 0.5) * h.width);
        }
    } else {
        double anchor = std::isfinite(d.lo) ? d.lo : d.mu;
        double w = (qhi - qlo) / 40;
        if (std::isfinite(d.lo) && std::isfinite(d.hi)) w = (d.hi - d.lo) / std::ceil((d.hi - d.lo) / w);
        h.width = w;
        long kmin = long(std::floor((qlo - anchor) / w));
        long kmax = long(std::ceil((qhi - anchor) / w));
        if (kmax <= kmin) kmax = kmin + 1;
        h.count.assign(kmax - kmin, 0);
        for (double v : x) {
            long k = long(std::floor((v - anchor) / w));
            if (k == kmax && v == d.hi) k = kmax - 1;
            if (k >= kmin && k < kmax) h.count[k - kmin]++;
        }
        for (long k = kmin; k < kmax; ++k) {
            h.left.push_back(anchor + k * w);
            h.right.push_back(anchor + (k + 1) * w);
        }
    }
    for (int c : h.count) h.dens.push_back(c / (double(R) * h.width));
    double a = h.left.front(), b = h.right.back(), pad = 0.05 * (b - a);
    h.plotLo = a - pad;
    h.plotHi = b + pad;
    return h;
}

// =====================================================================
// SVG drawing
// =====================================================================
const double PW = 420, PH = 330;
const double HEADER = 100;
const double ML = 60, MR = 18, MT = 52, MB = 46;

const char* C_BAR = "#7fb0dc";
const char* C_NORMAL = "#d1495b";
const char* C_POP = "#1f7a6d";
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

struct Panel {
    std::string title, info, xlabel, ylabel;
    double xlo, xhi, ymax;
    const Hist* hist = nullptr;
    std::vector<std::pair<double, double>> normal;   // N(mu, sigma^2/n) curve
    std::vector<std::pair<double, double>> pdf;      // population pdf
    std::vector<std::pair<double, double>> stems;    // population pmf
};

void drawPanel(std::ostringstream& o, const Panel& p, double px, double py, int id) {
    double x0 = px + ML, x1 = px + PW - MR, y0 = py + PH - MB, y1 = py + MT;
    auto sx = [&](double x) { return x0 + (x - p.xlo) / (p.xhi - p.xlo) * (x1 - x0); };
    auto sy = [&](double y) { return y0 - std::min(y, p.ymax) / p.ymax * (y0 - y1); };

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
    for (auto& [x, y] : p.stems) {
        o << fmt("<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"%s\" stroke-width=\"2.2\"/>\n",
                 sx(x), y0, sx(x), sy(y), C_POP);
        o << fmt("<circle cx=\"%.2f\" cy=\"%.2f\" r=\"4\" fill=\"%s\"/>\n", sx(x), sy(y), C_POP);
    }
    if (!p.normal.empty()) {
        std::ostringstream pts;
        for (auto& [x, y] : p.normal) pts << fmt("%.2f,%.2f ", sx(x), sy(y));
        o << "<polyline points=\"" << pts.str() << "\" fill=\"none\" stroke=\"" << C_NORMAL << "\" stroke-width=\"2\"/>\n";
    }
    o << "</g>\n";

    o << fmt("<polyline points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"none\" stroke=\"%s\"/>\n",
             x0, y1, x0, y0, x1, y0, C_SUB);
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"15\" font-weight=\"600\" fill=\"%s\">%s</text>\n",
             x0, py + 22, C_TEXT, p.title.c_str());
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12\" fill=\"%s\">%s</text>\n", x0, py + 40, C_SUB, p.info.c_str());
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12\" text-anchor=\"middle\" fill=\"%s\">%s</text>\n",
             (x0 + x1) / 2, y0 + 36, C_TEXT, p.xlabel.c_str());
    o << fmt("<text x=\"%.1f\" y=\"%.1f\" font-size=\"12\" text-anchor=\"middle\" fill=\"%s\" "
             "transform=\"rotate(-90 %.1f %.1f)\">%s</text>\n",
             px + 16, (y0 + y1) / 2, C_TEXT, px + 16, (y0 + y1) / 2, p.ylabel.c_str());
}

std::vector<std::pair<double, double>> normalCurve(double mu, double var, double lo, double hi) {
    std::vector<std::pair<double, double>> c;
    for (int i = 0; i <= 400; ++i) {
        double x = lo + (hi - lo) * i / 400;
        c.push_back({x, normalPdf(x, mu, var)});
    }
    return c;
}

// Extra points at the support bounds keep vertical edges (e.g. Uniform) sharp.
std::vector<std::pair<double, double>> pdfCurve(const Dist& d, double lo, double hi) {
    std::vector<double> xs;
    for (int i = 0; i <= 600; ++i) xs.push_back(lo + (hi - lo) * i / 600);
    for (double b : {d.lo, d.hi})
        if (std::isfinite(b) && b > lo && b < hi) {
            double eps = (hi - lo) * 1e-6;
            xs.push_back(b - eps); xs.push_back(b); xs.push_back(b + eps);
        }
    std::sort(xs.begin(), xs.end());
    std::vector<std::pair<double, double>> c;
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
    x += 34 + 7.0 * text.size();
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
    fs::path outDir = exeDir(argv) / "output_1-1";
    fs::create_directories(outDir / "svg");
    fs::create_directories(outDir / "csv");

    const auto dists = makeDists();
    const std::string XBAR = "x\u0304";

    std::ofstream summary(outDir / "csv" / "summary.csv");
    summary << "distribution,n,mean_sim,mean_theory,var_sim,var_theory\n";

    std::printf("SQC HW1 #1-1  CLT simulation   R = %d, seed = %u\n", R, SEED);
    std::printf("Theory: E(x\u0304) = \u03BC,  Var(x\u0304) = \u03C3\u00B2/n\n");

    for (unsigned di = 0; di < dists.size(); ++di) {
        const Dist& d = dists[di];
        std::vector<std::vector<double>> xbars;
        std::vector<Hist> hists;
        std::vector<Panel> panels(1 + N_LIST.size());

        for (int n : N_LIST) {
            xbars.push_back(simulateMeans(d, n, di));
            hists.push_back(makeHist(d, n, xbars.back()));
        }

        std::printf("\n%s   \u03BC = %g, \u03C3\u00B2 = %g\n", d.label.c_str(), d.mu, d.var);
        // header padded by hand: printf counts bytes, not display width, for x-bar / mu / sigma^2
        std::printf("      n |      mean(x\u0304)            \u03BC |       var(x\u0304)         \u03C3\u00B2/n\n");
        std::printf("  ------+---------------------------+--------------------------\n");

        // panels 2-6: x-bar for each n
        std::ofstream hcsv(outDir / "csv" / ("hist_" + d.key + ".csv"));
        hcsv << "n,bin_left,bin_right,bin_mid,count,density,normal_pdf\n";
        for (size_t j = 0; j < N_LIST.size(); ++j) {
            int n = N_LIST[j];
            const Hist& h = hists[j];
            Stats st = describe(xbars[j]);
            double tv = d.var / n;

            Panel& p = panels[j + 1];
            p.title = fmt("n = %d", n);
            p.info = fmt("mean %.4f (\u03BC = %.4g)  \u00B7  var %.4g (\u03C3\u00B2/n = %.4g)", st.mean, d.mu, st.var, tv);
            p.xlabel = XBAR;
            p.ylabel = "density";
            p.xlo = h.plotLo; p.xhi = h.plotHi;
            p.hist = &h;
            p.normal = normalCurve(d.mu, tv, p.xlo, p.xhi);
            double ymax = *std::max_element(h.dens.begin(), h.dens.end());
            p.ymax = std::max(ymax, normalPdf(d.mu, d.mu, tv)) * 1.1;

            for (size_t i = 0; i < h.dens.size(); ++i) {
                double mid = (h.left[i] + h.right[i]) / 2;
                hcsv << n << ',' << h.left[i] << ',' << h.right[i] << ',' << mid << ',' << h.count[i] << ','
                     << h.dens[i] << ',' << normalPdf(mid, d.mu, tv) << '\n';
            }
            summary << d.key << ',' << n << ',' << st.mean << ',' << d.mu << ',' << st.var << ',' << tv << '\n';
            std::printf("  %5d | %12.5f %12.5f | %12.6f %12.6f\n", n, st.mean, d.mu, st.var, tv);
        }

        // panel 1: theoretical population, same x range as n = 1
        Panel& pop = panels[0];
        pop.title = "Population: " + d.label;
        pop.info = fmt("\u03BC = %.4g  \u00B7  \u03C3\u00B2 = %.4g", d.mu, d.var);
        pop.xlabel = "x";
        pop.ylabel = d.discrete ? "P(X = x)" : "f(x)";
        pop.xlo = hists[0].plotLo; pop.xhi = hists[0].plotHi;
        std::ofstream pcsv(outDir / "csv" / ("population_" + d.key + ".csv"));
        pcsv << (d.discrete ? "x,pmf\n" : "x,pdf\n");
        double ymax = 0;
        if (d.discrete) {
            for (long k = long(std::ceil(pop.xlo)); k <= long(std::floor(pop.xhi)); ++k) {
                double y = d.f(double(k));
                pop.stems.push_back({double(k), y});
                pcsv << k << ',' << y << '\n';
                ymax = std::max(ymax, y);
            }
        } else {
            pop.pdf = pdfCurve(d, pop.xlo, pop.xhi);
            for (auto& [x, y] : pop.pdf) {
                pcsv << x << ',' << y << '\n';
                if (std::isfinite(y)) ymax = std::max(ymax, y);
            }
        }
        pop.ymax = ymax * 1.1;

        std::ofstream rcsv(outDir / "csv" / ("raw_" + d.key + ".csv"));
        rcsv << "rep";
        for (int n : N_LIST) rcsv << ",n=" << n;
        rcsv << '\n';
        for (int r = 0; r < R; ++r) {
            rcsv << r + 1;
            for (size_t j = 0; j < N_LIST.size(); ++j) rcsv << ',' << xbars[j][r];
            rcsv << '\n';
        }

        double W = 3 * PW + 20, H = HEADER + 2 * PH + 16;
        std::ostringstream o;
        o << svgHeader(W, H);
        o << fmt("<text x=\"30\" y=\"36\" font-size=\"21\" font-weight=\"700\" fill=\"%s\">"
                 "CLT simulation: sampling distribution of %s \u2014 %s</text>\n",
                 C_TEXT, XBAR.c_str(), d.label.c_str());
        o << fmt("<text x=\"30\" y=\"59\" font-size=\"12.5\" fill=\"%s\">R = %d replications per n, seed = %u. "
                 "Each panel uses its own axis range so the shape stays readable.</text>\n",
                 C_SUB, R, SEED);
        double lx = 30, ly = 82;
        legendItem(o, lx, ly, fmt("<rect x=\"%.1f\" y=\"%.1f\" width=\"20\" height=\"11\" fill=\"%s\"/>\n", lx, ly - 5, C_BAR),
                   "simulated " + XBAR + " (histogram, density)");
        legendItem(o, lx, ly, fmt("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"2.2\"/>\n",
                                  lx, ly, lx + 20, ly, C_NORMAL),
                   "normal approximation N(\u03BC, \u03C3\u00B2/n)");
        legendItem(o, lx, ly, fmt("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"2.2\"/>\n",
                                  lx, ly, lx + 20, ly, C_POP),
                   d.discrete ? "population pmf" : "population pdf");
        for (size_t i = 0; i < panels.size(); ++i) {
            double px = 10 + (i % 3) * PW, py = HEADER + (i / 3) * PH;
            drawPanel(o, panels[i], px, py, int(i));
        }
        o << "</svg>\n";
        std::ofstream(outDir / "svg" / (d.key + ".svg")) << o.str();
    }

    std::cout << "\nOutput written to " << (exeDir(argv).filename() / "output_1-1").u8string() << '\n';

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
