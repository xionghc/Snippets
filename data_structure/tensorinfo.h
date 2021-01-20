#include <numeric>
#include <vector>


enum class DataType {
  // fixed point types
  UINT8 = 0,
  INT8,
  UINT16,
  INT16,
  INT32,
  INT64,
  UINT32,
  UINT64,
  BOOL,
  // floating point types
  FLOAT,
  FLOAT16,
  BFLOAT16,
  DOUBLE,
  COMPLEX64,
  COMPLEX128,
  // other types
  STRING,
  UNDEFINED,
};

// Type alias to make clearer.
using Shape = std::vector<int64_t>;
using Rank = int;


class DataTypeInfo;

class TensorInfo {
public:
  /// Create TensorInformation based on data type and shape
  ///
  /// \param data_type    - The data type
  /// \param shape        - The actual shape of the tensor
  TensorInfo(DataType, const Shape &);
  /// Create TensorInformation based on data type, shape and meta shape
  ///
  /// \param data_type    - The data type
  /// \param shape        - The actual shape of the tensor
  /// \param meta_shape   - The meta shape of the tensor, which can for example
  ///                       be used to store the original tensor shape before
  ///                       replicated tensor sharding was applied
  TensorInfo(DataType data_type, const Shape &shape, const Shape &meta_shape);
  TensorInfo(std::string data_type, std::string shape);
  TensorInfo(std::string data_type, const Shape &);

  TensorInfo() = default;
  void set(DataType, const Shape &);
  void set(DataType, const Shape &, const Shape &);
  const Shape &shape() const;
  //
  const Shape &meta_shape() const;
  // A helper functions for back-ends which
  // prefer the size as (unsigned) size_t.
  std::vector<size_t> shape_szt() const;
  // Defined in-header to encourage inlining (it's called a lot).
  Rank rank() const { return static_cast<int>(shape_v.size()); }
  // Defined in-header to encourage inlining (it's called a lot).
  int64_t nelms() const {
    return std::accumulate(shape_v.begin(),
                           shape_v.end(),
                           static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
  }
  // total bytes of tensor
  int64_t nbytes() const;
  // Defined in-header to encourage inlining (it's called a lot).
  int64_t dim(int i) const {
    if (i >= shape_v.size()) {
      throw "Invalid input dimension {}, tensor of rank {}";
    }
    return shape_v[i];
  }
  DataType dataType() const;
  const std::string &data_type() const;
  const std::string &data_type_lcase() const;
  void append(std::ostream &) const;
  bool isSet() const;
  bool operator==(const TensorInfo &) const;
  bool operator!=(const TensorInfo &) const;

private:
  // The tensor's actual shape
  Shape shape_v;
  // The tensor's meta shape, e.g. original shape before replicated tensor
  // sharding
  Shape meta_shape_v;

};
