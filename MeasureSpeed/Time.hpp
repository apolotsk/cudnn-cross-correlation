#pragma once
#include <stdint.h> // For int64_t.
#include <time.h> // For timespec, clock_nanosleep(), etc.
#include <sys/time.h> // For gettimeofday().

/* System time */
struct SystemTime {
  timeval t;
  SystemTime() { gettimeofday(&t, NULL); }
  inline operator timeval() { return t; }
  double Timestamp() const { return t.tv_sec + t.tv_usec/1e6; }
};

template<clockid_t clock_id>
struct Time {
  timespec t;
  Time() { clock_gettime(clock_id, &t); }
  Time(double seconds) {
    int64_t nanoseconds = int64_t((seconds-int(seconds))*1e9);
    t = {int(seconds), nanoseconds};
  }
  inline operator timespec() { return t; }
  double Timestamp() const { return t.tv_sec + t.tv_nsec/1e9; }
  void Wait() {
    clock_nanosleep(clock_id, TIMER_ABSTIME, &t, NULL);
  }
};

/* System-wide realtime clock. */
typedef Time<CLOCK_REALTIME> RealTime;

/* Monotonic time since some unspecified starting point. */
typedef Time<CLOCK_MONOTONIC> MonotonicTime;

/* All threads all measured and summed up. */
typedef Time<CLOCK_PROCESS_CPUTIME_ID> CpuTime;

/* Only main thread is mesured. */
typedef Time<CLOCK_THREAD_CPUTIME_ID> ThreadTime;

/* All threads are measured and summed up. */
struct ClockTickTime {
  clock_t t;
  ClockTickTime() { t = clock(); }
  inline operator clock_t() { return t; }
  double Timestamp() const { return (double)t/CLOCKS_PER_SEC; }
};

inline double timestamp() { return RealTime().Timestamp(); }
