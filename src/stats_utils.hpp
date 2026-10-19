// stats_utils.hpp
// ---------------------------------------------------------------
// 統計工具函式：
//   1. 樣本動差（平均數、標準差、偏態、超額峰態）
//   2. 分位數、直方圖（含離散分配的格點對齊）
//   3. 常態與卡方分配的 pdf / cdf / 分位數
//   4. 核密度估計 KDE（把模擬資料畫成平滑的密度曲線）
//   5. Kolmogorov–Smirnov 距離（衡量模擬分布與理論分布的差距）
// ---------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// ---------- 1. 樣本動差 ----------
struct Moments {
    double mean = 0, sd = 0, skew = 0, exkurt = 0;
};

// 計算平均數、標準差（除以 R−1）、偏態 g1、超額峰態 g2
inline Moments computeMoments(const std::vector<double>& v) {
    Moments m;
    const double R = static_cast<double>(v.size());
    for (double x : v) m.mean += x;
    m.mean /= R;
    double m2 = 0, m3 = 0, m4 = 0;  // 中央動差
    for (double x : v) {
        double d = x - m.mean, d2 = d * d;
        m2 += d2; m3 += d2 * d; m4 += d2 * d2;
    }
    m2 /= R; m3 /= R; m4 /= R;
    m.sd = std::sqrt(m2 * R / (R - 1));
    m.skew = m2 > 0 ? m3 / std::pow(m2, 1.5) : 0.0;
    m.exkurt = m2 > 0 ? m4 / (m2 * m2) - 3.0 : 0.0;
    return m;
}

// ---------- 2. 分位數與直方圖 ----------

// 已排序資料的第 p 分位數（線性內插，與 Excel PERCENTILE.INC 相同）
inline double quantileSorted(const std::vector<double>& s, double p) {
    double pos = p * (s.size() - 1);
    size_t i = static_cast<size_t>(pos);
    if (i + 1 >= s.size()) return s.back();
    double frac = pos - i;
    return s[i] + frac * (s[i + 1] - s[i]);
}

struct Histogram {
    double lo = 0, width = 1;      // 第一個 bin 的左邊界、bin 寬度
    std::vector<int> counts;
    int under = 0, over = 0;       // 落在顯示範圍外的個數
};

// 建立直方圖。
//   [tLo, tHi]  : 理論上希望涵蓋的範圍（例如 μ ± 4σ/√n）
//   step        : 離散格點間距（X̄ 只會是 k/n → step = 1/n）；連續分配傳 0
//   targetBins  : 目標 bin 數
// 離散分配時，bin 邊界對齊在格點中間、寬度取格點間距的整數倍，
// 否則每個 bin 包含的格點數不同，直方圖會出現規律的「梳子狀」缺口。
inline Histogram makeHistogram(const std::vector<double>& sorted, double tLo, double tHi,
                               double step, int targetBins) {
    Histogram h;
    double dataMin = sorted.front(), dataMax = sorted.back();
    // 顯示範圍：理論範圍與資料 0.05%~99.95% 分位數的聯集，但不超出資料實際範圍
    double lo = std::max(dataMin, std::min(tLo, quantileSorted(sorted, 0.0005)));
    double hi = std::min(dataMax, std::max(tHi, quantileSorted(sorted, 0.9995)));
    if (hi <= lo) hi = lo + (step > 0 ? step : 1e-9);

    double w = (hi - lo) / targetBins;
    if (step > 0) {
        lo = (std::round(lo / step) - 0.5) * step;          // 邊界放在格點中間
        w = step * std::max(1.0, std::round(w / step));     // 寬度 = 格點間距的整數倍
    }
    // 離散：hi 是格點，會落在某個 bin 的中間 → floor + 1 個 bin
    // 連續：用 ceil，並把恰好等於 hi 的值歸入最後一個 bin（否則最後一個 bin 只會裝到最大值，幾乎是空的）
    int nb = step > 0 ? static_cast<int>(std::floor((hi - lo) / w)) + 1
                      : std::max(1, static_cast<int>(std::ceil((hi - lo) / w - 1e-9)));
    h.lo = lo; h.width = w;
    h.counts.assign(nb, 0);
    for (double x : sorted) {
        int idx = static_cast<int>(std::floor((x - lo) / w));
        if (step == 0 && idx == nb && x <= hi) idx = nb - 1;
        if (idx < 0) ++h.under;
        else if (idx >= nb) ++h.over;
        else ++h.counts[idx];
    }
    return h;
}

// ---------- 3. 常態與卡方分配 ----------
const double PI = 3.14159265358979323846;

inline double normalPdf(double x, double mu, double sd) {
    double z = (x - mu) / sd;
    return std::exp(-0.5 * z * z) / (sd * std::sqrt(2 * PI));
}

inline double normalCdf(double x, double mu, double sd) {
    return 0.5 * std::erfc(-(x - mu) / (sd * std::sqrt(2.0)));
}

// 正規化下不完全 Gamma 函數 P(a, x)（Numerical Recipes 的級數 / 連分式法）
inline double gammaP(double a, double x) {
    if (x <= 0) return 0.0;
    double logPrefix = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1) {  // 級數展開
        double ap = a, del = 1.0 / a, sum = del;
        for (int i = 0; i < 10000; ++i) {
            ap += 1; del *= x / ap; sum += del;
            if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
        }
        return sum * std::exp(logPrefix);
    }
    // 連分式（Lentz 法）求 Q(a, x)，再取 1 − Q
    const double tiny = 1e-300;
    double b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
    for (int i = 1; i < 10000; ++i) {
        double an = -i * (i - a);
        b += 2;
        d = an * d + b; if (std::fabs(d) < tiny) d = tiny;
        c = b + an / c; if (std::fabs(c) < tiny) c = tiny;
        d = 1 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1) < 1e-15) break;
    }
    return 1.0 - std::exp(logPrefix) * h;
}

// χ²_k 的 pdf 與 cdf（χ²_k ≡ Gamma(α = k/2, β = 2)）
inline double chiSquarePdf(double x, double k) {
    if (x <= 0) return 0.0;
    double a = k / 2;
    return std::exp((a - 1) * std::log(x) - x / 2 - a * std::log(2.0) - std::lgamma(a));
}

inline double chiSquareCdf(double x, double k) { return gammaP(k / 2, x / 2); }

// 以二分法求遞增函數 cdf 的反函數（分位數）
inline double invertCdf(const std::function<double(double)>& cdf, double p, double lo, double hi) {
    for (int i = 0; i < 200; ++i) {
        double mid = 0.5 * (lo + hi);
        (cdf(mid) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

// ---------- 4. 核密度估計（KDE）：把模擬資料畫成平滑的密度曲線 ----------
// f̂(x) = 1/(R·h) Σ K((x − xi)/h)，K 為標準常態核函數。

// Silverman 經驗法則頻寬：h = 0.9 · min(s, IQR/1.34) · R^(−1/5)
// 離散母體的 X̄ 只落在間距 step 的格點上，頻寬至少取 step，曲線才不會變成一根根尖刺。
inline double kdeBandwidth(const std::vector<double>& sorted, double sd, double step) {
    double iqr = quantileSorted(sorted, 0.75) - quantileSorted(sorted, 0.25);
    double spread = iqr > 0 ? std::min(sd, iqr / 1.34) : sd;
    double h = 0.9 * spread * std::pow(static_cast<double>(sorted.size()), -0.2);
    return std::max({h, step, 1e-12});
}

// 計算 Σ K((c − xi)/h)：已排序資料只需加總 c ± 5h 範圍內的點（範圍外貢獻 < 1e-6）
inline double kernelSum(const std::vector<double>& sorted, double c, double h) {
    auto first = std::lower_bound(sorted.begin(), sorted.end(), c - 5 * h);
    auto last = std::upper_bound(first, sorted.end(), c + 5 * h);
    double sum = 0;
    for (auto it = first; it != last; ++it) {
        double u = (c - *it) / h;
        sum += std::exp(-0.5 * u * u);
    }
    return sum;
}

// 在 x 處的 KDE 值。
// [lower, upper] 為資料的支撐範圍（例如指數分配 X̄ ≥ 0）。在邊界做「反射」修正：
// 否則邊界附近有一半的機率質量會漏到範圍外，密度被低估約一半。
inline double kdeDensity(const std::vector<double>& sorted, double h, double x,
                         double lower = -INFINITY, double upper = INFINITY) {
    double sum = kernelSum(sorted, x, h);
    if (std::isfinite(lower)) sum += kernelSum(sorted, 2 * lower - x, h);
    if (std::isfinite(upper)) sum += kernelSum(sorted, 2 * upper - x, h);
    return sum / (sorted.size() * h * std::sqrt(2 * PI));
}

// ---------- 5. Kolmogorov–Smirnov 距離 ----------
// D = max |F_模擬(x) − F_理論(x)|，越接近 0 代表越吻合。
// 離散分配（step > 0）使用連續性修正：在格點 x 上比較 F_理論(x ± step/2)，
// 否則階梯狀的經驗 cdf 與連續的常態 cdf 必然有一段跳躍差，會高估 D。
inline double ksDistance(const std::vector<double>& sorted, const std::function<double(double)>& cdf,
                         double step) {
    const double R = static_cast<double>(sorted.size());
    double D = 0;
    size_t i = 0;
    while (i < sorted.size()) {
        size_t j = i;
        while (j < sorted.size() && sorted[j] == sorted[i]) ++j;  // 處理重複值（離散分配）
        double x = sorted[i];
        double Fbefore = i / R, Fafter = j / R;                   // x 左右兩側的經驗 cdf
        double Fleft = cdf(x - step / 2), Fright = cdf(x + step / 2);
        D = std::max({D, std::fabs(Fafter - Fright), std::fabs(Fbefore - Fleft)});
        i = j;
    }
    return D;
}
