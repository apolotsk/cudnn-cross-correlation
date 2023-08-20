#pragma once
#include <chrono> // For `std::chrono`.

struct SteadyClockTime {
  std::chrono::steady_clock::time_point time_point;
  SteadyClockTime() { time_point = std::chrono::steady_clock::now(); }
  inline operator std::chrono::steady_clock::time_point() { return time_point; }
  double Timestamp() const { return std::chrono::duration<double>(time_point.time_since_epoch()).count(); }
};

class Stopwatch {
  typedef SteadyClockTime TimeStruct;
  TimeStruct time;
public:
  /* Returns seconds since the last Split() or Stopwatch(), and resets timer. */
  double Time() const { return TimeStruct().Timestamp()-time.Timestamp(); }
  double Split() {
    double seconds_elapsed = Time();
    Reset();
    return seconds_elapsed;
  }
  void Reset() { time = TimeStruct(); }
};
