// main.cpp
// ===============================================================
// SQC HW1：以模擬驗證
//   #1-1 中央極限定理（CLT）：樣本平均數 X̄ 的抽樣分配 → N(μ, σ²/n)
//   #1-2 樣本變異數的抽樣分配：(n−1)S²/σ² ~ χ²_{n−1}（常態母體）
//
// 模擬設計：
//   n = 每組樣本的大小（1 ~ 100，每個整數都模擬）
//   R = 重複抽樣次數（預設 10000）→ 每個 n 得到 R 個 X̄（或 S²）
//   每種分配提供數組預設參數（見 makeFamilies），每一組參數都完整模擬一次。
//   每一組（題目, 參數組, n）各自用獨立的種子初始化亂數，
//   因此結果可重現，且不受執行順序（多執行緒）影響。
//
// 輸出：
//   output/excel/                 給 Excel 作圖用的 CSV（UTF-8 BOM，避免符號亂碼）
//   web/data.js                   網頁的索引資料（分配清單、參數、趨勢圖用的摘要）
//   web/data/{clt,s2}/<參數組>.js  每組參數的詳細資料，網頁選到時才載入
//
// 編譯：g++ -std=c++17 -O2 -o sim src/main.cpp
// 執行：./sim [seed] [R]      例：./sim 20261016 10000
// ===============================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "distributions.hpp"
#include "stats_utils.hpp"

namespace fs = std::filesystem;

// ---------------- 模擬參數 ----------------
struct Config {
    std::uint64_t seed = 20261016;  // 亂數種子（報告中請註明）
    int R = 10000;                  // 每個 n 重複抽樣的次數
    int nMax = 100;                 // 樣本數 n 的上限
    int bins = 50;                  // 直方圖目標 bin 數
    int curvePoints = 121;          // 理論曲線 / KDE 取樣點數
    // 匯出給 Excel 的 n（全部匯出會超過 Excel 列數上限且不實用）
    std::vector<int> excelN_clt = {1, 2, 5, 10, 30, 100};
    std::vector<int> excelN_s2 = {2, 3, 5, 10, 30, 100};
};

// ---------------- 分配家族與預設參數組 ----------------
struct ParamSpec {
    std::string key;            // 檔名用的 ASCII 代號
    std::string label;          // 顯示用符號
    std::vector<double> values; // 可選的參數值
    int def;                    // 預設值的索引
};

struct Family {
    std::string id;
    std::string name;
    Kind kind;
    bool discrete;
    std::vector<ParamSpec> params;  // 參數順序與 makePopulation(kind, a, b) 的 a、b 相同
    bool inClt = true;              // false = 只用在 #1-2
};

// 參數值的選擇原則：涵蓋「接近對稱」到「高度偏斜」，讓收斂速度的差異看得出來
std::vector<Family> makeFamilies() {
    return {
        // ---- 離散型 ----
        {"bernoulli", "Bernoulli", Kind::Bernoulli, true, {{"p", "p", {0.05, 0.2, 0.5}, 0}}},
        {"binomial", "Binomial", Kind::Binomial, true, {{"m", "m", {5, 10, 30}, 1}, {"p", "p", {0.05, 0.2, 0.5}, 1}}},
        {"poisson", "Poisson", Kind::Poisson, true, {{"lambda", "λ", {0.5, 1, 2, 5, 10}, 2}}},
        // ---- 連續型 ----
        {"normal", "Normal", Kind::Normal, false, {{"mu", "μ", {0, 10}, 1}, {"sigma", "σ", {1, 2, 5}, 1}}},
        {"uniform", "Uniform", Kind::Uniform, false, {{"a", "a", {-2, 0}, 1}, {"b", "b", {1, 4}, 0}}},
        {"exponential", "Exponential", Kind::Exponential, false, {{"beta", "β", {0.5, 1, 2}, 1}}},
        {"gamma", "Gamma", Kind::Gamma, false, {{"alpha", "α", {0.5, 2, 5}, 1}, {"beta", "β", {1, 2}, 1}}},
        {"lognormal", "Lognormal", Kind::Lognormal, false, {{"mu", "μ", {0, 1}, 0}, {"sigma", "σ", {0.25, 0.5, 1}, 2}}},
        // #1-2 專用：對稱但厚尾（超額峰態 6/(ν−4)），ν 越大越接近常態 → 偏離 χ² 的程度可調
        {"t", "t", Kind::StudentT, false, {{"nu", "ν", {5, 10, 30}, 1}}, false},
    };
}

// 一組具體參數（例如 Binomial(m=10, p=0.2)）
struct Variant {
    size_t fam;
    std::vector<int> choice;     // 每個參數選第幾個值
    std::vector<double> values;
    bool isDefault;
    std::string key;             // 例：binomial_m10_p0.2
    Population pop;
};

std::string fmtG(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", x);
    return buf;
}

// 列舉每個家族所有參數值的組合（笛卡兒積）
std::vector<Variant> enumerateVariants(const std::vector<Family>& fams) {
    std::vector<Variant> out;
    for (size_t f = 0; f < fams.size(); ++f) {
        const auto& ps = fams[f].params;
        std::vector<int> c(ps.size(), 0);
        while (true) {
            Variant v;
            v.fam = f; v.choice = c; v.isDefault = true; v.key = fams[f].id;
            for (size_t i = 0; i < ps.size(); ++i) {
                v.values.push_back(ps[i].values[c[i]]);
                v.isDefault = v.isDefault && c[i] == ps[i].def;
                v.key += "_" + ps[i].key + fmtG(ps[i].values[c[i]]);
            }
            v.pop = makePopulation(fams[f].kind, v.values[0], v.values.size() > 1 ? v.values[1] : 0.0);
            out.push_back(v);
            size_t i = 0;  // 進位到下一個組合
            while (i < ps.size() && ++c[i] == static_cast<int>(ps[i].values.size())) c[i++] = 0;
            if (i == ps.size()) break;
        }
    }
    return out;
}

// 由參數組代號產生穩定的雜湊值，當作亂數種子的一部分（新增參數組不會影響其他組的結果）
unsigned stableHash(const std::string& s) {
    std::uint32_t h = 2166136261u;  // FNV-1a
    for (unsigned char ch : s) { h ^= ch; h *= 16777619u; }
    return h;
}

// 每組（題目, 參數組, n）各自建立獨立的亂數產生器
Engine makeEngine(std::uint64_t seed, unsigned task, unsigned variantHash, unsigned n) {
    std::seed_seq seq{static_cast<unsigned>(seed), static_cast<unsigned>(seed >> 32), task, variantHash, n};
    return Engine(seq);
}

// ---------------- 輸出格式小工具 ----------------
// 數字 → 字串（NaN/Inf 轉為 null）；網頁畫圖用的陣列只需 4 位有效數字，可大幅縮小檔案
std::string num(double x, int prec = 6) {
    if (!std::isfinite(x)) return "null";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*g", prec, x);
    return buf;
}

template <class T>
std::string arr(const std::vector<T>& v, int prec = 6) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        s += num(static_cast<double>(v[i]), prec);
    }
    return s + "]";
}

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

const char* BOM = "\xEF\xBB\xBF";  // UTF-8 BOM：讓 Excel 正確辨識 μ、σ 等符號

void writeFile(const fs::path& path, const std::string& content, bool bom = false) {
    std::ofstream f(path, std::ios::binary);
    if (bom) f << BOM;
    f << content;
}

bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

// 在直方圖範圍內等距取樣曲線
struct Curve {
    double x0 = 0, dx = 0;
    std::vector<double> y;
};

template <class Pdf>
Curve sampleCurve(double lo, double hi, int points, Pdf pdf) {
    Curve c;
    c.x0 = lo;
    c.dx = (hi - lo) / (points - 1);
    for (int i = 0; i < points; ++i) c.y.push_back(pdf(c.x0 + i * c.dx));
    return c;
}

template <class Pdf>
Curve sampleCurve(const Histogram& h, int points, Pdf pdf) {
    return sampleCurve(h.lo, h.lo + h.width * h.counts.size(), points, pdf);
}

std::string curveJson(const Curve& c) {
    return "{\"x0\":" + num(c.x0) + ",\"dx\":" + num(c.dx) + ",\"y\":" + arr(c.y, 4) + "}";
}

std::string histJson(const Histogram& h) {
    return "\"lo\":" + num(h.lo) + ",\"w\":" + num(h.width) + ",\"counts\":" + arr(h.counts) +
           ",\"under\":" + std::to_string(h.under) + ",\"over\":" + std::to_string(h.over);
}

// 把直方圖逐 bin 寫成 CSV（含模擬密度、理論密度與 KDE，方便 Excel 畫「直條 + 折線」組合圖）
template <class Pdf, class Kde>
void writeHistCsv(std::ostream& f, int n, const Histogram& h, int R, Pdf pdf, Kde kde) {
    for (size_t i = 0; i < h.counts.size(); ++i) {
        double lo = h.lo + i * h.width, hi = lo + h.width, mid = 0.5 * (lo + hi);
        f << n << ',' << num(lo) << ',' << num(hi) << ',' << num(mid) << ',' << h.counts[i] << ','
          << num(h.counts[i] / (R * h.width)) << ',' << num(pdf(mid)) << ',' << num(kde(mid)) << '\n';
    }
}

// 原始資料寫成寬格式 CSV：每一欄是一個 n，每一列是一次重複抽樣
std::string rawCsv(const std::vector<std::string>& headers, const std::vector<std::vector<double>>& cols, int R) {
    std::ostringstream f;
    f << "rep";
    for (const auto& h : headers) f << ',' << h;
    f << '\n';
    for (int r = 0; r < R; ++r) {
        f << r + 1;
        for (const auto& col : cols) f << ',' << num(col[r]);
        f << '\n';
    }
    return f.str();
}

// Q-Q 圖使用的機率點 p = 0.01, 0.02, ..., 0.99
std::vector<double> qqProbs() {
    std::vector<double> p;
    for (int i = 1; i <= 99; ++i) p.push_back(i / 100.0);
    return p;
}

// 母體分配圖：連續 → pdf 曲線；離散 → pmf 各點
std::string popJson(const Config& cfg, const Variant& v) {
    const Population& pop = v.pop;
    if (pop.lattice) {
        std::vector<double> xs, ys;
        for (int k = pop.pmfLo; k <= pop.pmfHi; ++k) { xs.push_back(k); ys.push_back(populationPmf(pop, k)); }
        return "{\"type\":\"pmf\",\"x\":" + arr(xs) + ",\"y\":" + arr(ys, 5) + "}";
    }
    // 先抽 20000 個樣本，以分位數決定合適的顯示範圍（有界的一端直接用支撐邊界）
    Engine g = makeEngine(cfg.seed, 0, stableHash(v.key), 0);
    std::vector<double> s(20000);
    withDistribution(pop, [&](auto dist) { for (double& x : s) x = static_cast<double>(dist(g)); });
    std::sort(s.begin(), s.end());
    double lo = std::isfinite(pop.supLo) ? pop.supLo : quantileSorted(s, 0.001);
    double hi = std::isfinite(pop.supHi) ? pop.supHi : quantileSorted(s, 0.999);
    double pad = 0.04 * (hi - lo);
    Curve c = sampleCurve(lo - pad, hi + pad, 161, [&](double x) { return populationPdf(pop, x); });
    return "{\"type\":\"pdf\",\"x0\":" + num(c.x0) + ",\"dx\":" + num(c.dx) + ",\"y\":" + arr(c.y, 4) + "}";
}

// 一組參數模擬完的所有輸出（先存在記憶體，最後由主執行緒依序寫檔）
struct JobResult {
    std::string detailJs;     // web/data/<task>/<key>.js
    std::string trendJson;    // 放進 data.js 的摘要（趨勢圖用）
    std::string summaryRows;  // 摘要 CSV 的資料列
    std::string histCsv;      // 選定 n 的直方圖 CSV
    std::string raw;          // 選定 n 的原始資料 CSV（只有預設參數組才輸出）
    double seconds = 0;
};

// ===============================================================
// #1-1 中央極限定理
// ===============================================================
JobResult runClt(const Config& cfg, const Variant& v) {
    const Population& pop = v.pop;
    const double sigma = std::sqrt(pop.var);
    const auto P = qqProbs();
    JobResult res;
    std::ostringstream byN, summary, hist;
    hist << "n,bin_lo,bin_hi,bin_mid,count,density_sim,density_normal_theory,density_kde\n";
    std::vector<double> tSkew, tKurt, tKs, tSdRatio, tSkewT, tKurtT;
    std::vector<std::vector<double>> excelRaw;
    std::vector<std::string> excelHeaders;

    for (int n = 1; n <= cfg.nMax; ++n) {
        // ---- 1. 抽樣：重複 R 次，每次抽 n 個觀測值，計算 X̄ 與 S ----
        std::vector<double> means(cfg.R);
        double sumS = 0;
        Engine g = makeEngine(cfg.seed, 1, stableHash(v.key), static_cast<unsigned>(n));
        withDistribution(pop, [&](auto dist) {
            std::vector<double> x(n);
            for (int r = 0; r < cfg.R; ++r) {
                double sum = 0;
                for (int i = 0; i < n; ++i) { x[i] = static_cast<double>(dist(g)); sum += x[i]; }
                double xbar = sum / n;
                means[r] = xbar;
                if (n > 1) {  // 樣本標準差 S（除以 n−1）
                    double ss = 0;
                    for (int i = 0; i < n; ++i) ss += (x[i] - xbar) * (x[i] - xbar);
                    sumS += std::sqrt(ss / (n - 1));
                }
            }
        });
        if (v.isDefault && contains(cfg.excelN_clt, n)) {
            excelRaw.push_back(means);
            excelHeaders.push_back("xbar_n" + std::to_string(n));
        }

        // ---- 2. 模擬統計量 vs. 理論值 ----
        // CLT：X̄ 的平均數 = μ、標準差 = σ/√n；偏態 = 母體偏態/√n、超額峰態 = 母體峰態/n
        Moments m = computeMoments(means);
        double meanS = n > 1 ? sumS / cfg.R : NAN;
        double se = sigma / std::sqrt(n);
        double skewT = pop.skew / std::sqrt(n), kurtT = pop.exkurt / n;

        std::vector<double> sorted = means;
        std::sort(sorted.begin(), sorted.end());
        double step = pop.lattice ? 1.0 / n : 0.0;  // 離散母體：X̄ 只會落在 k/n
        auto normalAt = [&](double x) { return normalPdf(x, pop.mean, se); };
        Histogram h = makeHistogram(sorted, pop.mean - 4 * se, pop.mean + 4 * se, step, cfg.bins);
        double ks = ksDistance(sorted, [&](double x) { return normalCdf(x, pop.mean, se); }, step);
        std::vector<double> qq;
        for (double p : P) qq.push_back(quantileSorted(sorted, p));
        Curve curve = sampleCurve(h, cfg.curvePoints, normalAt);
        // 模擬資料的 KDE 曲線（與理論曲線取在相同的 x 點上，方便直接對照）
        double bw = kdeBandwidth(sorted, m.sd, step);
        auto kdeAt = [&](double x) { return kdeDensity(sorted, bw, x, pop.supLo, pop.supHi); };
        Curve kde = sampleCurve(h, cfg.curvePoints, kdeAt);

        tSkew.push_back(m.skew); tKurt.push_back(m.exkurt); tKs.push_back(ks);
        tSdRatio.push_back(m.sd / se); tSkewT.push_back(skewT); tKurtT.push_back(kurtT);

        // ---- 3. 輸出 ----
        summary << v.key << ',' << pop.id << ',' << quoted(pop.label) << ',' << n << ',' << num(m.mean) << ','
                << num(pop.mean) << ',' << num(m.sd) << ',' << num(se) << ',' << (n > 1 ? num(meanS) : "") << ','
                << num(sigma) << ',' << num(m.skew) << ',' << num(skewT) << ',' << num(m.exkurt) << ','
                << num(kurtT) << ',' << num(ks) << '\n';
        if (contains(cfg.excelN_clt, n)) writeHistCsv(hist, n, h, cfg.R, normalAt, kdeAt);

        if (n > 1) byN << ',';
        byN << "{\"n\":" << n << ',' << histJson(h)
            << ",\"sim\":{\"mean\":" << num(m.mean) << ",\"sd\":" << num(m.sd) << ",\"meanS\":" << num(meanS)
            << ",\"skew\":" << num(m.skew) << ",\"exkurt\":" << num(m.exkurt) << ",\"ks\":" << num(ks) << '}'
            << ",\"theo\":{\"mean\":" << num(pop.mean) << ",\"sd\":" << num(se) << ",\"skew\":" << num(skewT)
            << ",\"exkurt\":" << num(kurtT) << '}' << ",\"curve\":" << curveJson(curve)
            << ",\"kde\":" << arr(kde.y, 4) << ",\"bw\":" << num(bw) << ",\"qq\":" << arr(qq, 5) << '}';
    }

    res.detailJs = "SIM_LOAD(\"clt\"," + quoted(v.key) + ",{\"pop\":" + popJson(cfg, v) + ",\"byN\":[" + byN.str() + "]});\n";
    res.trendJson = "{\"skew\":" + arr(tSkew, 4) + ",\"exkurt\":" + arr(tKurt, 4) + ",\"ks\":" + arr(tKs, 4) +
                    ",\"sdratio\":" + arr(tSdRatio, 4) + ",\"tskew\":" + arr(tSkewT, 4) +
                    ",\"texkurt\":" + arr(tKurtT, 4) + "}";
    res.summaryRows = summary.str();
    res.histCsv = hist.str();
    if (v.isDefault) res.raw = rawCsv(excelHeaders, excelRaw, cfg.R);
    return res;
}

// ===============================================================
// #1-2 樣本變異數 S² 的抽樣分配
// ===============================================================
JobResult runS2(const Config& cfg, const Variant& v) {
    const Population& pop = v.pop;
    const double sigma2 = pop.var;
    const auto P = qqProbs();
    JobResult res;
    std::ostringstream byN, summary, hist;
    hist << "n,bin_lo,bin_hi,bin_mid,count,density_sim,density_chi2_theory,density_kde\n";
    std::vector<double> tVar, tTrueVar, tSkew, tKurt, tKs, tSkewT, tKurtT;
    std::vector<std::vector<double>> rawS2, rawChi;
    std::vector<std::string> hS2, hChi;

    for (int n = 2; n <= cfg.nMax; ++n) {  // n = 1 時 S² 無定義（自由度 0）
        const double k = n - 1;              // 自由度

        // ---- 1. 抽樣：每次抽 n 個，計算 S² 與 χ² 統計量 (n−1)S²/σ² ----
        std::vector<double> s2(cfg.R), chi(cfg.R);
        Engine g = makeEngine(cfg.seed, 2, stableHash(v.key), static_cast<unsigned>(n));
        withDistribution(pop, [&](auto dist) {
            std::vector<double> x(n);
            for (int r = 0; r < cfg.R; ++r) {
                double sum = 0;
                for (int i = 0; i < n; ++i) { x[i] = static_cast<double>(dist(g)); sum += x[i]; }
                double xbar = sum / n, ss = 0;
                for (int i = 0; i < n; ++i) ss += (x[i] - xbar) * (x[i] - xbar);
                s2[r] = ss / (n - 1);   // S² = Σ(Xi − X̄)² / (n−1)
                chi[r] = ss / sigma2;   // (n−1)S²/σ² = Σ(Xi − X̄)² / σ²
            }
        });
        if (v.isDefault && contains(cfg.excelN_s2, n)) {
            rawS2.push_back(s2); rawChi.push_back(chi);
            hS2.push_back("S2_n" + std::to_string(n)); hChi.push_back("chi_n" + std::to_string(n));
        }

        // ---- 2. 模擬統計量 vs. 理論值 ----
        // 常態母體：χ²_k 的平均數 k、變異數 2k、偏態 √(8/k)、超額峰態 12/k
        // 任意母體的真實變異數：Var = 2k + κ·k²/n（κ 為母體超額峰態），用來說明非常態為何偏離 χ²
        Moments mc = computeMoments(chi), ms = computeMoments(s2);
        double sdT = std::sqrt(2 * k), skewT = std::sqrt(8 / k), kurtT = 12 / k;
        double trueVar = 2 * k + pop.exkurt * k * k / n;

        std::vector<double> sorted = chi;
        std::sort(sorted.begin(), sorted.end());
        // 整數值母體：n·Σ(Xi − X̄)² 必為整數 → χ² 統計量只落在 1/(n·σ²) 的格點上
        double step = pop.lattice ? 1.0 / (n * sigma2) : 0.0;
        auto chiAt = [&](double x) { return chiSquarePdf(x, k); };
        Histogram h = makeHistogram(sorted, std::max(0.0, k - 4 * sdT), k + 4 * sdT, step, cfg.bins);
        double ks = ksDistance(sorted, [&](double x) { return chiSquareCdf(x, k); }, step);
        std::vector<double> qq, qqT;
        double upper = k + 50 * sdT + 50;
        for (double p : P) {
            qq.push_back(quantileSorted(sorted, p));
            qqT.push_back(invertCdf([&](double x) { return chiSquareCdf(x, k); }, p, 0.0, upper));
        }
        Curve curve = sampleCurve(h, cfg.curvePoints, chiAt);
        // 模擬資料的 KDE 曲線；(n−1)S²/σ² ≥ 0，在 0 做邊界反射修正（離散母體不做，理由同 CLT）
        double bw = kdeBandwidth(sorted, mc.sd, step);
        auto kdeAt = [&](double x) { return kdeDensity(sorted, bw, x, pop.lattice ? -INFINITY : 0.0); };
        Curve kde = sampleCurve(h, cfg.curvePoints, kdeAt);

        tVar.push_back(mc.sd * mc.sd / (2 * k)); tTrueVar.push_back(trueVar / (2 * k));
        tSkew.push_back(mc.skew); tKurt.push_back(mc.exkurt); tKs.push_back(ks);
        tSkewT.push_back(skewT); tKurtT.push_back(kurtT);

        // ---- 3. 輸出 ----
        summary << v.key << ',' << pop.id << ',' << quoted(pop.label) << ',' << n << ',' << k << ','
                << num(ms.mean) << ',' << num(sigma2) << ',' << num(ms.sd * ms.sd) << ','
                << num(2 * sigma2 * sigma2 / k) << ',' << num(mc.mean) << ',' << num(k) << ','
                << num(mc.sd * mc.sd) << ',' << num(2 * k) << ',' << num(trueVar) << ',' << num(mc.skew) << ','
                << num(skewT) << ',' << num(mc.exkurt) << ',' << num(kurtT) << ',' << num(ks) << '\n';
        if (contains(cfg.excelN_s2, n)) writeHistCsv(hist, n, h, cfg.R, chiAt, kdeAt);

        if (n > 2) byN << ',';
        byN << "{\"n\":" << n << ",\"df\":" << k << ',' << histJson(h)
            << ",\"sim\":{\"mean\":" << num(mc.mean) << ",\"sd\":" << num(mc.sd) << ",\"skew\":" << num(mc.skew)
            << ",\"exkurt\":" << num(mc.exkurt) << ",\"ks\":" << num(ks) << ",\"meanS2\":" << num(ms.mean)
            << ",\"varS2\":" << num(ms.sd * ms.sd) << '}'
            << ",\"theo\":{\"mean\":" << num(k) << ",\"sd\":" << num(sdT) << ",\"skew\":" << num(skewT)
            << ",\"exkurt\":" << num(kurtT) << ",\"trueSd\":" << num(std::sqrt(trueVar)) << '}'
            << ",\"curve\":" << curveJson(curve) << ",\"kde\":" << arr(kde.y, 4) << ",\"bw\":" << num(bw)
            << ",\"qq\":" << arr(qq, 5) << ",\"qqT\":" << arr(qqT, 5) << '}';
    }

    res.detailJs = "SIM_LOAD(\"s2\"," + quoted(v.key) + ",{\"pop\":" + popJson(cfg, v) + ",\"byN\":[" + byN.str() + "]});\n";
    res.trendJson = "{\"var\":" + arr(tVar, 4) + ",\"truevar\":" + arr(tTrueVar, 4) + ",\"skew\":" + arr(tSkew, 4) +
                    ",\"exkurt\":" + arr(tKurt, 4) + ",\"ks\":" + arr(tKs, 4) + ",\"tskew\":" + arr(tSkewT, 4) +
                    ",\"texkurt\":" + arr(tKurtT, 4) + "}";
    res.summaryRows = summary.str();
    res.histCsv = hist.str();
    if (v.isDefault) {
        std::vector<std::string> headers = hS2;
        headers.insert(headers.end(), hChi.begin(), hChi.end());
        std::vector<std::vector<double>> cols = rawS2;
        cols.insert(cols.end(), rawChi.begin(), rawChi.end());
        res.raw = rawCsv(headers, cols, cfg.R);
    }
    return res;
}

// ===============================================================
// 網頁索引資料 data.js：分配清單、參數組、各參數組的趨勢摘要
// ===============================================================
std::string catalogJson(const std::vector<Family>& fams, const std::vector<Variant>& vars) {
    std::ostringstream js;
    js << "\"families\":[";
    for (size_t f = 0; f < fams.size(); ++f) {
        const Family& F = fams[f];
        if (f) js << ',';
        js << "{\"id\":" << quoted(F.id) << ",\"name\":" << quoted(F.name)
           << ",\"discrete\":" << (F.discrete ? "true" : "false")
           << ",\"tabs\":" << (F.inClt ? "[\"clt\",\"s2\"]" : "[\"s2\"]") << ",\"params\":[";
        for (size_t i = 0; i < F.params.size(); ++i) {
            const ParamSpec& p = F.params[i];
            if (i) js << ',';
            js << "{\"key\":" << quoted(p.key) << ",\"label\":" << quoted(p.label) << ",\"values\":" << arr(p.values)
               << ",\"def\":" << p.def << '}';
        }
        js << "]}";
    }
    js << "],\"variants\":{";
    for (size_t i = 0; i < vars.size(); ++i) {
        const Variant& v = vars[i];
        const Population& p = v.pop;
        if (i) js << ',';
        js << quoted(v.key) << ":{\"family\":" << quoted(fams[v.fam].id) << ",\"label\":" << quoted(p.label)
           << ",\"values\":" << arr(v.values) << ",\"mean\":" << num(p.mean) << ",\"var\":" << num(p.var)
           << ",\"sd\":" << num(std::sqrt(p.var)) << ",\"skew\":" << num(p.skew) << ",\"exkurt\":" << num(p.exkurt)
           << '}';
    }
    js << '}';
    return js.str();
}

// ===============================================================
int main(int argc, char* argv[]) {
    Config cfg;
    if (argc > 1) cfg.seed = std::stoull(argv[1]);
    if (argc > 2) cfg.R = std::stoi(argv[2]);

    const auto fams = makeFamilies();
    const auto vars = enumerateVariants(fams);

    fs::path excelDir = fs::path("output") / "excel";
    fs::path webDir = "web";
    for (const char* sub : {"clt_hist", "clt_raw", "s2_hist", "s2_raw"}) fs::create_directories(excelDir / sub);
    fs::create_directories(webDir / "data" / "clt");
    fs::create_directories(webDir / "data" / "s2");

    // ---- 工作清單：每個參數組 × 兩個題目，以多執行緒平行模擬 ----
    struct Job { int task; size_t vi; };
    std::vector<Job> jobs;
    for (int task : {1, 2})
        for (size_t i = 0; i < vars.size(); ++i)
            if (task == 2 || fams[vars[i].fam].inClt) jobs.push_back({task, i});
    std::vector<JobResult> results(jobs.size());

    unsigned nThreads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "seed = " << cfg.seed << ", R = " << cfg.R << ", n = 1.." << cfg.nMax << ", "
              << vars.size() << " parameter sets, " << jobs.size() << " jobs, " << nThreads << " threads\n";
    auto t0 = std::chrono::steady_clock::now();

    std::atomic<size_t> next{0};
    std::mutex ioMutex;
    size_t done = 0;
    auto worker = [&]() {
        for (size_t j; (j = next++) < jobs.size();) {
            auto t1 = std::chrono::steady_clock::now();
            const Variant& v = vars[jobs[j].vi];
            results[j] = jobs[j].task == 1 ? runClt(cfg, v) : runS2(cfg, v);
            results[j].seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            std::lock_guard<std::mutex> lock(ioMutex);
            std::printf("  [%3zu/%zu] %-4s %-28s %5.1f s\n", ++done, jobs.size(), jobs[j].task == 1 ? "CLT" : "S2",
                        v.key.c_str(), results[j].seconds);
            std::fflush(stdout);
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nThreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    // ---- 依固定順序寫出所有檔案 ----
    std::string cltSummary =
        "variant,distribution,label,n,mean_xbar_sim,mean_xbar_theory,sd_xbar_sim,sd_xbar_theory,"
        "mean_S_sim,sigma_population,skew_sim,skew_theory,exkurt_sim,exkurt_theory,KS\n";
    std::string s2Summary =
        "variant,population,label,n,df,mean_S2_sim,mean_S2_theory,var_S2_sim,var_S2_theory,"
        "mean_chi_sim,mean_chi_theory,var_chi_sim,var_chi_theory,var_chi_true,"
        "skew_sim,skew_theory,exkurt_sim,exkurt_theory,KS\n";
    std::ostringstream trendClt, trendS2;
    for (size_t j = 0; j < jobs.size(); ++j) {
        const Variant& v = vars[jobs[j].vi];
        const JobResult& r = results[j];
        bool clt = jobs[j].task == 1;
        std::string task = clt ? "clt" : "s2";
        writeFile(webDir / "data" / task / (v.key + ".js"), r.detailJs);
        writeFile(excelDir / (task + "_hist") / (v.key + ".csv"), r.histCsv, true);
        if (!r.raw.empty()) writeFile(excelDir / (task + "_raw") / (v.key + ".csv"), r.raw, true);
        (clt ? cltSummary : s2Summary) += r.summaryRows;
        std::ostringstream& tr = clt ? trendClt : trendS2;
        if (tr.tellp() > 0) tr << ',';
        tr << quoted(v.key) << ':' << r.trendJson;
    }
    writeFile(excelDir / "clt_summary.csv", cltSummary, true);
    writeFile(excelDir / "s2_summary.csv", s2Summary, true);

    // 網頁索引資料：寫成 JS 檔而不是 JSON，直接雙擊 index.html 開啟時瀏覽器才不會擋掉
    std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    std::vector<double> qqZ;  // 標準常態分位數，網頁畫 CLT 的 Q-Q 圖用
    for (double p : qqProbs()) qqZ.push_back(invertCdf([](double z) { return normalCdf(z, 0, 1); }, p, -10, 10));

    std::ostringstream out;
    out << "// 由 src/main.cpp 自動產生，請勿手動修改\n"
        << "window.SIM_DATA = {\"meta\":{\"seed\":" << cfg.seed << ",\"R\":" << cfg.R << ",\"nMax\":" << cfg.nMax
        << ",\"generated\":" << quoted(stamp) << ",\"qqP\":" << arr(qqProbs()) << ",\"qqZ\":" << arr(qqZ) << "},"
        << catalogJson(fams, vars) << ",\"trend\":{\"clt\":{" << trendClt.str() << "},\"s2\":{" << trendS2.str()
        << "}}};\n";
    writeFile(webDir / "data.js", out.str());

    auto sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "All done in " << sec << " s\n"
              << "  Excel CSV : " << excelDir.string() << "\n"
              << "  Web data  : " << (webDir / "data.js").string() << " + " << (webDir / "data").string() << "\n";
    return 0;
}
