#include "tensorinfo.h"

// A class hold the tensor data.
class TensorData {
 public:
  TensorData(const TensorInfo & , const void *src);

  void resetData(const TensorInfo &, const void *src);

  template <typename RESULT_TYPE>
  std::vector<RESULT_TYPE> copyDataAs(int expectedResultSize) const {
    if (data_.size() != expectedResultSize * sizeof(RESULT_TYPE)) {
      throw "Size of data does not match expected result size.";
    }

    std::vector<RESULT_TYPE> result;
    const RESULT_TYPE *x = reinterpret_cast<RESULT_TYPE *>(data());

    for (size_t i = 0; i < expectedResultSize; ++i) {
      result.push_back(x[i]);
    }

    return result;
  }

  void *data();
  const void *data() const;

 private:
  // Using a vector of char(1 byte) to hold the data_.
  std::vector<char> data_;
}


class Tensor{

}
