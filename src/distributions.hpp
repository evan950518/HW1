// distributions.hpp
// ---------------------------------------------------------------
// 母體分配的定義：參數、理論動差、機率密度（pdf / pmf），
// 以及把母體對應到 C++ <random> 分配物件的轉接函式。
//
// 參數化一律採用課本（Walpole）的寫法：
//   Exp(β)      : β 為平均數（C++ 需要的是速率 λ = 1/β）
//   Gamma(α, β) : β 為尺度參數（與 C++ 相同）
//   Lognormal(μ, σ²) : ln X ~ N(μ, σ²)
// ---------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using Engine = std::mt19937_64;  // 亂數產生器（64 位元 Mersenne Twister）

enum class Kind { Normal, Uniform, Binomial, Poisson, Exponential, Gamma, Lognormal, Bernoulli, StudentT };

struct Population {
    std::string id;      // 程式內部代號（也用於檔名）
    std::string label;   // 顯示用名稱（含參數）
    Kind kind;
    double a, b;         // 分配參數，意義依 kind 而定（見 makePopulation）
    bool lattice;        // 是否只取整數值（離散分配）→ X̄ 只會落在 k/n 上
    double mean, var;    // 母體平均數 μ、變異數 σ²
    double skew;         // 母體偏態係數
    double exkurt;       // 母體超額峰態（常態 = 0）
    int pmfLo, pmfHi;    // 離散分配用：pmf 的顯示範圍
    double supLo = -INFINITY, supHi = INFINITY;  // 連續分配的支撐範圍（X̄ 也落在此範圍內），供 KDE 邊界修正
};

// 依種類與參數建立母體，同時算好理論動差
inline Population makePopulation(Kind kind, double a, double b = 0.0) {
    Population p{};
    p.kind = kind; p.a = a; p.b = b; p.lattice = false;
    char buf[128];
    switch (kind) {
    case Kind::Normal:       // a = μ, b = σ
        p.id = "normal";
        std::snprintf(buf, sizeof buf, "Normal(μ=%g, σ=%g)", a, b);
        p.mean = a; p.var = b * b; p.skew = 0; p.exkurt = 0;
        break;
    case Kind::Uniform:      // a = 下界, b = 上界
        p.id = "uniform";
        std::snprintf(buf, sizeof buf, "Uniform(a=%g, b=%g)", a, b);
        p.mean = (a + b) / 2; p.var = (b - a) * (b - a) / 12; p.skew = 0; p.exkurt = -1.2;
        p.supLo = a; p.supHi = b;
        break;
    case Kind::Binomial: {   // a = 試驗次數 m, b = 成功機率 p
        p.id = "binomial"; p.lattice = true;
        std::snprintf(buf, sizeof buf, "Binomial(m=%g, p=%g)", a, b);
        double q = 1 - b;
        p.mean = a * b; p.var = a * b * q;
        p.skew = (1 - 2 * b) / std::sqrt(p.var);
        p.exkurt = (1 - 6 * b * q) / p.var;
        // pmf 只畫到機率不可忽略處（例如 Binomial(30, 0.05) 不必畫到 x = 30）
        p.pmfLo = 0; p.pmfHi = std::min(static_cast<int>(a), static_cast<int>(std::ceil(p.mean + 6 * std::sqrt(p.var) + 3)));
        break;
    }
    case Kind::Poisson:      // a = λ
        p.id = "poisson"; p.lattice = true;
        std::snprintf(buf, sizeof buf, "Poisson(λ=%g)", a);
        p.mean = a; p.var = a; p.skew = 1 / std::sqrt(a); p.exkurt = 1 / a;
        p.pmfLo = 0; p.pmfHi = static_cast<int>(std::ceil(a + 6 * std::sqrt(a) + 3));
        break;
    case Kind::Exponential:  // a = β（平均數）
        p.id = "exponential";
        std::snprintf(buf, sizeof buf, "Exponential(β=%g)", a);
        p.mean = a; p.var = a * a; p.skew = 2; p.exkurt = 6;
        p.supLo = 0;
        break;
    case Kind::Gamma:        // a = α（形狀）, b = β（尺度）
        p.id = "gamma";
        std::snprintf(buf, sizeof buf, "Gamma(α=%g, β=%g)", a, b);
        p.mean = a * b; p.var = a * b * b; p.skew = 2 / std::sqrt(a); p.exkurt = 6 / a;
        p.supLo = 0;
        break;
    case Kind::Lognormal: {  // a = μ, b = σ（皆為 ln X 的參數）
        p.id = "lognormal";
        std::snprintf(buf, sizeof buf, "Lognormal(μ=%g, σ=%g)", a, b);
        double s2 = b * b, e = std::exp(s2);
        p.mean = std::exp(a + s2 / 2);
        p.var = (e - 1) * std::exp(2 * a + s2);
        p.skew = (e + 2) * std::sqrt(e - 1);
        p.exkurt = std::exp(4 * s2) + 2 * std::exp(3 * s2) + 3 * std::exp(2 * s2) - 6;
        p.supLo = 0;
        break;
    }
    case Kind::Bernoulli:    // a = p
        p.id = "bernoulli"; p.lattice = true;
        std::snprintf(buf, sizeof buf, "Bernoulli(p=%g)", a);
        p.mean = a; p.var = a * (1 - a);
        p.skew = (1 - 2 * a) / std::sqrt(p.var);
        p.exkurt = (1 - 6 * a * (1 - a)) / p.var;
        p.pmfLo = 0; p.pmfHi = 1;
        break;
    case Kind::StudentT:     // a = 自由度 ν（需 ν > 4，峰態才有限）
        p.id = "t";
        std::snprintf(buf, sizeof buf, "t(ν=%g)", a);
        p.mean = 0; p.var = a / (a - 2); p.skew = 0; p.exkurt = 6 / (a - 4);
        break;
    }
    p.label = buf;
    return p;
}

// 依母體種類建立對應的 C++ <random> 分配物件，並交給 f 使用。
// 用泛型 lambda 接收，讓內層抽樣迴圈對每種分配都能被編譯器最佳化。
template <class F>
void withDistribution(const Population& p, F&& f) {
    switch (p.kind) {
    case Kind::Normal:      f(std::normal_distribution<double>(p.a, p.b)); break;
    case Kind::Uniform:     f(std::uniform_real_distribution<double>(p.a, p.b)); break;
    case Kind::Binomial:    f(std::binomial_distribution<int>(static_cast<int>(p.a), p.b)); break;
    case Kind::Poisson:     f(std::poisson_distribution<int>(p.a)); break;
    case Kind::Exponential: f(std::exponential_distribution<double>(1.0 / p.a)); break;  // C++ 參數為速率 λ = 1/β
    case Kind::Gamma:       f(std::gamma_distribution<double>(p.a, p.b)); break;
    case Kind::Lognormal:   f(std::lognormal_distribution<double>(p.a, p.b)); break;
    case Kind::Bernoulli:   f(std::bernoulli_distribution(p.a)); break;
    case Kind::StudentT:    f(std::student_t_distribution<double>(p.a)); break;
    }
}

// 連續分配的 pdf（離散分配請用 populationPmf）
inline double populationPdf(const Population& p, double x) {
    const double PI = 3.14159265358979323846;
    switch (p.kind) {
    case Kind::Normal: {
        double z = (x - p.a) / p.b;
        return std::exp(-0.5 * z * z) / (p.b * std::sqrt(2 * PI));
    }
    case Kind::Uniform:
        return (x >= p.a && x <= p.b) ? 1.0 / (p.b - p.a) : 0.0;
    case Kind::Exponential:
        return x < 0 ? 0.0 : std::exp(-x / p.a) / p.a;
    case Kind::Gamma:
        if (x <= 0) return 0.0;
        return std::exp((p.a - 1) * std::log(x) - x / p.b - std::lgamma(p.a) - p.a * std::log(p.b));
    case Kind::Lognormal: {
        if (x <= 0) return 0.0;
        double z = (std::log(x) - p.a) / p.b;
        return std::exp(-0.5 * z * z) / (x * p.b * std::sqrt(2 * PI));
    }
    case Kind::StudentT: {   // Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν+1)/2)
        double nu = p.a;
        return std::exp(std::lgamma((nu + 1) / 2) - std::lgamma(nu / 2) - 0.5 * std::log(nu * PI) -
                        (nu + 1) / 2 * std::log(1 + x * x / nu));
    }
    default:
        return 0.0;
    }
}

// 離散分配的 pmf：P(X = k)
inline double populationPmf(const Population& p, int k) {
    switch (p.kind) {
    case Kind::Binomial: {
        int m = static_cast<int>(p.a);
        if (k < 0 || k > m) return 0.0;
        double logC = std::lgamma(m + 1.0) - std::lgamma(k + 1.0) - std::lgamma(m - k + 1.0);
        return std::exp(logC + k * std::log(p.b) + (m - k) * std::log(1 - p.b));
    }
    case Kind::Poisson:
        if (k < 0) return 0.0;
        return std::exp(k * std::log(p.a) - p.a - std::lgamma(k + 1.0));
    case Kind::Bernoulli:
        return k == 1 ? p.a : (k == 0 ? 1 - p.a : 0.0);
    default:
        return 0.0;
    }
}
