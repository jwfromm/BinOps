// binarize.h
#ifndef BINARIZE_H_
#define BINARIZE_H_

template <typename Device, typename T>
struct BinarizeFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#endif
