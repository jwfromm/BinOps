// multibit2.h
#ifndef MULTIBIT2_H_
#define MULTIBIT2_H_

template <typename Device, typename T>
struct Multibit2Functor {
  void operator()(const Device& d, int size, const int *b, const T* in, T* out);
};

#endif
