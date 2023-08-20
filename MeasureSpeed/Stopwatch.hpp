#pragma once
#include "Time.hpp"

class Stopwatch {
  typedef MonotonicTime TimeStruct;
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
