#pragma once
#include <time.h>    // For srand().
#include <inttypes.h> // For uint8_t, uint32_t, etc.
#include <stdlib.h>

typedef __fp16 half;

inline void rand_init() { srand((uint32_t)time(NULL)); }

template <typename T> T rand();
template<> inline bool rand<bool>() { return ::rand()&1; }
template<> inline uint8_t rand<uint8_t>() { return ::rand()&0xFF; }
template<> inline int8_t rand<int8_t>() { return ::rand()&0xFF; }
template<> inline float rand<float>() { return (float)::rand()/RAND_MAX; }
template<> inline half rand<half>() { return (half)::rand()/RAND_MAX; }

template <typename T>
inline void* rand(void* data, int count) {
  for (int i = 0; i<count; ++i) ((T*)data)[i] = rand<T>();
  return data;
}
