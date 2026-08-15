// C++ comparison implementation of the pora_rs hot loop, bound with
// pybind11. Same numeric path as src/lib.rs step for step - including the
// exactness-preserving Gaussian tail cutoff - so the differential tests
// hold it to the same 1e-12 bar as the Rust extension. Single-threaded by
// design: the point of this file is a language comparison on identical
// semantics, not a parallelism study.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double PI = 3.14159265358979323846;

using Foe = std::tuple<double, double, double, double, double, double>;
using Ego =
    std::tuple<double, double, double, double, double, double, double, double>;

double ssd(double speed_ms, double reaction_time_s, double decel_ms2) {
  return speed_ms * reaction_time_s +
         speed_ms * speed_ms / (2.0 * decel_ms2);
}

struct BoxDims {
  double hx_out, hy_out, hx_in, hy_in;

  static BoxDims derive(double av_length, double av_width,
                        double fleet_max_length, double fleet_min_width,
                        double speed_ms, double reaction_time_s,
                        double decel_ms2) {
    const double phi_length =
        av_length + fleet_max_length + ssd(speed_ms, reaction_time_s, decel_ms2);
    const double phi_width = av_width + fleet_max_length;
    const double core_length = std::min(av_length + fleet_min_width, phi_length);
    const double core_width = std::min(av_width + fleet_min_width, phi_width);
    return BoxDims{phi_length / 2.0, phi_width / 2.0, core_length / 2.0,
                   core_width / 2.0};
  }

  double pcgo(double x, double y) const {
    if (std::abs(x) > hx_out || std::abs(y) > hy_out) return 0.0;
    const double dx = hx_out - hx_in;
    const double dy = hy_out - hy_in;
    const double tx = dx <= 0.0 ? 0.0 : (std::abs(x) - hx_in) / dx;
    const double ty = dy <= 0.0 ? 0.0 : (std::abs(y) - hy_in) / dy;
    const double t = std::clamp(std::max(tx, ty), 0.0, 1.0);
    return 1.0 - t;
  }
};

double bilinear(const std::vector<double>& data, std::int64_t ny,
                std::int64_t nx, double ox, double oy, double res, double x,
                double y) {
  const double gx = (x - ox) / res;
  const double gy = (y - oy) / res;
  const auto i0 = static_cast<std::int64_t>(std::floor(gx));
  const auto j0 = static_cast<std::int64_t>(std::floor(gy));
  const double fx = gx - static_cast<double>(i0);
  const double fy = gy - static_cast<double>(j0);
  double out = 0.0;
  const std::tuple<std::int64_t, std::int64_t, double> corners[4] = {
      {i0, j0, (1.0 - fx) * (1.0 - fy)},
      {i0 + 1, j0, fx * (1.0 - fy)},
      {i0, j0 + 1, (1.0 - fx) * fy},
      {i0 + 1, j0 + 1, fx * fy},
  };
  for (const auto& [ii, jj, w] : corners) {
    if (ii >= 0 && ii < nx && jj >= 0 && jj < ny) {
      out += data[static_cast<std::size_t>(jj * nx + ii)] * w;
    }
  }
  return out;
}

// Constant-velocity Gaussian occupancy (parity: cv_gaussian_impl).
void cv_gaussian(const std::vector<Foe>& foes, double lead_time, double ox,
                 double oy, std::size_t ny, std::size_t nx, double res,
                 double sigma0, double sigma_growth, std::vector<double>& out) {
  constexpr double P_CUT = 1e-20;

  struct Pre {
    double mx, my, two_s2, footprint, cutoff_d2;
  };
  std::vector<Pre> pre;
  pre.reserve(foes.size());
  for (const auto& f : foes) {
    const double fx = std::get<0>(f), fy = std::get<1>(f);
    const double fvx = std::get<2>(f), fvy = std::get<3>(f);
    const double fl = std::get<4>(f), fw = std::get<5>(f);
    const double mx = fx + fvx * lead_time;
    const double my = fy + fvy * lead_time;
    const double sigma = sigma0 + sigma_growth * lead_time + (fl + fw) / 8.0;
    const double footprint = fl * fw;
    const double two_s2 = 2.0 * sigma * sigma;
    const double peak = footprint / (PI * two_s2);
    if (peak < P_CUT) continue;  // below cutoff everywhere
    const double cutoff_d2 = two_s2 * std::log(peak / P_CUT);
    pre.push_back({mx, my, two_s2, footprint, cutoff_d2});
  }

  for (std::size_t j = 0; j < ny; ++j) {
    const double y = oy + res * static_cast<double>(j);
    for (std::size_t i = 0; i < nx; ++i) {
      const double x = ox + res * static_cast<double>(i);
      double free = 1.0;
      for (const auto& p : pre) {
        const double d2 =
            (x - p.mx) * (x - p.mx) + (y - p.my) * (y - p.my);
        if (d2 > p.cutoff_d2) continue;
        const double density = std::exp(-d2 / p.two_s2) / (PI * p.two_s2);
        const double occ = std::clamp(density * p.footprint, 0.0, 1.0);
        free *= 1.0 - occ;
      }
      out[j * nx + i] = 1.0 - free;
    }
  }
}

std::size_t window_len(double half, double res) {
  const auto n = static_cast<std::int64_t>(std::llround(2.0 * half / res)) + 1;
  return static_cast<std::size_t>(std::max<std::int64_t>(1, n));
}

struct StepOut {
  std::vector<double> po;
  double adj_max;
};

StepOut pora_step(const std::vector<double>& global, std::int64_t gny,
                  std::int64_t gnx, double gox, double goy, double gres,
                  double av_x, double av_y, double heading, double half_length,
                  double half_width, double res, const BoxDims& bx,
                  const std::vector<double>* prev_po, double beta) {
  const std::size_t nx = window_len(half_length, res);
  const std::size_t ny = window_len(half_width, res);
  const double c = std::cos(heading);
  const double s = std::sin(heading);
  const double exp_beta = std::exp(beta);

  std::vector<double> po(ny * nx, 0.0);
  double adj_max = -std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < ny; ++j) {
    const double yl = -half_width + res * static_cast<double>(j);
    for (std::size_t i = 0; i < nx; ++i) {
      const double xl = -half_length + res * static_cast<double>(i);
      const double wx = av_x + c * xl - s * yl;
      const double wy = av_y + s * xl + c * yl;
      const double p_occ = bilinear(global, gny, gnx, gox, goy, gres, wx, wy);
      po[j * nx + i] = p_occ;
      const double pc = bx.pcgo(xl, yl) * p_occ;
      double a;
      if (prev_po == nullptr) {
        a = pc;
      } else {
        const double d_po = p_occ - (*prev_po)[j * nx + i];
        a = pc * std::exp(beta * d_po) / exp_beta;
      }
      adj_max = std::max(adj_max, a);
    }
  }
  return {std::move(po), adj_max};
}

double pora_score(const Ego& ego, const std::vector<Foe>& foes,
                  double horizon_s, double dt, double beta, double resolution,
                  double extent, double reaction_time_s, double decel_ms2,
                  double sigma0, double sigma_growth) {
  if (foes.empty()) return 0.0;

  const auto& [ex, ey, evx, evy, heading, speed, el, ew] = ego;
  double fleet_max_l = -std::numeric_limits<double>::infinity();
  double fleet_min_w = std::numeric_limits<double>::infinity();
  for (const auto& f : foes) {
    fleet_max_l = std::max(fleet_max_l, std::get<4>(f));
    fleet_min_w = std::min(fleet_min_w, std::get<5>(f));
  }

  const auto n = static_cast<std::size_t>(
                     std::llround(2.0 * extent / resolution)) + 1;
  const double ox = ex - extent;
  const double oy = ey - extent;
  const auto steps =
      static_cast<std::size_t>(std::llround(horizon_s / dt)) + 1;

  const BoxDims bx = BoxDims::derive(el, ew, fleet_max_l, fleet_min_w, speed,
                                     reaction_time_s, decel_ms2);
  const double half_len = bx.hx_out + 1.0;
  const double half_wid = bx.hy_out + 1.0;

  std::vector<double> global(n * n, 0.0);
  std::vector<double> prev_po;
  bool have_prev = false;
  double scalar = -std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < steps; ++k) {
    const double t = static_cast<double>(k) * dt;
    cv_gaussian(foes, t, ox, oy, n, n, resolution, sigma0, sigma_growth,
                global);
    StepOut out = pora_step(global, static_cast<std::int64_t>(n),
                            static_cast<std::int64_t>(n), ox, oy, resolution,
                            ex + evx * t, ey + evy * t, heading, half_len,
                            half_wid, resolution, bx,
                            have_prev ? &prev_po : nullptr, beta);
    scalar = std::max(scalar, out.adj_max);
    prev_po = std::move(out.po);
    have_prev = true;
  }
  return scalar;
}

}  // namespace

PYBIND11_MODULE(pora_cpp, m) {
  m.doc() = "C++ comparison implementation of the pora_rs hot loop";
  m.def("pora_score", &pora_score, py::arg("ego"), py::arg("foes"),
        py::arg("horizon_s") = 2.5, py::arg("dt") = 0.5, py::arg("beta") = 1.0,
        py::arg("resolution") = 0.5, py::arg("extent") = 40.0,
        py::arg("reaction_time_s") = 1.0, py::arg("decel_ms2") = 9.81,
        py::arg("sigma0") = 0.5, py::arg("sigma_growth") = 0.5,
        py::call_guard<py::gil_scoped_release>());
}
