# NumPy Notes with Brief Examples and Code

## 1️⃣ Basics and Setup
- **Purpose**: NumPy is a fundamental package for numerical computations in Python, enabling efficient array operations for data analysis.
- **Installation**: `pip install numpy`
- **Import**: `import numpy as np`

**Code Example**:
```python
import numpy as np
# Check NumPy version
print(np.__version__)
```

## 2️⃣ NumPy Arrays
- **Creating Arrays**:
  - `np.array()`: Create array from list/tuple
  - `np.zeros()`, `np.ones()`: Arrays filled with 0s or 1s
  - `np.arange(start, stop, step)`: Range of values
  - `np.linspace(start, stop, num)`: Evenly spaced values
  - `np.random.*`: Random arrays
- **Properties**: `shape`, `dtype`, `size`, `ndim`
- **Reshaping**: `reshape()`, `ravel()`, `flatten()`
- **Indexing/Slicing**: Access elements like Python lists
- **Boolean/Fancy Indexing**: Select with conditions or indices
- **Copy vs. View**: `copy()` creates new array; `view()` shares memory

**Code Example**:
```python
# Create arrays
arr = np.array([1, 2, 3, 4])
zeros = np.zeros((2, 3))
arange = np.arange(0, 10, 2)
random = np.random.rand(3)

print("Array:", arr)
print("Zeros:\n", zeros)
print("Arange:", arange)
print("Random:", random)
print("Shape:", arr.shape, "Dtype:", arr.dtype, "Ndim:", arr.ndim)

# Reshape
reshaped = arr.reshape(2, 2)
print("Reshaped:\n", reshaped)

# Indexing
print("First element:", arr[0])
print("Slice:", arr[1:3])

# Boolean indexing
print("Values > 2:", arr[arr > 2])
```

## 3️⃣ Array Operations
- **Element-wise**: `+`, `-`, `*`, `/`
- **Broadcasting**: Operate on arrays of different shapes
- **Universal Functions (ufuncs)**: `np.sqrt()`, `np.exp()`, `np.log()`
- **Aggregation**: `sum()`, `mean()`, `std()`, `min()`, `max()`, `argmin()`, `argmax()`
- **Axis-based**: Apply operations along specific axes

**Code Example**:
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
print("Add:", a + b)
print("Multiply:", a * b)

# Broadcasting
print("Add scalar:", a + 2)

# Ufuncs
print("Sqrt:", np.sqrt(a))

# Aggregation
print("Sum:", np.sum(a))
print("Mean along axis 0:", np.mean([[1, 2], [3, 4]], axis=0))
```

## 4️⃣ Advanced Array Manipulation
- **Transpose**: `.T` or `np.transpose()`
- **Concatenation**: `np.concatenate()`, `np.vstack()`, `np.hstack()`
- **Splitting**: `np.split()`, `np.hsplit()`, `np.vsplit()`
- **Stacking**: Combine arrays along new axis
- **Tile/Repeat**: Replicate arrays
- **Sorting**: `np.sort()`, `np.argsort()`
- **Unique**: `np.unique()`

**Code Example**:
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# Transpose
print("Transpose:\n", a.T)

# Concatenation
print("Vstack:\n", np.vstack((a, b)))

# Sorting
arr = np.array([3, 1, 4, 2])
print("Sorted:", np.sort(arr))
print("Indices of sort:", np.argsort(arr))

# Unique
print("Unique:", np.unique([1, 2, 2, 3]))
```

## 5️⃣ Random Module
- **Seed**: `np.random.seed()` for reproducibility
- **Sampling**: `rand()` (uniform [0,1)), `randn()` (normal), `randint()`, `choice()`
- **Distributions**: Normal, binomial, uniform, etc.
- **Shuffling**: `np.random.shuffle()`

**Code Example**:
```python
np.random.seed(42)
print("Random uniform:", np.random.rand(3))
print("Random integers:", np.random.randint(1, 10, 3))
print("Choice:", np.random.choice([1, 2, 3, 4], size=2))

# Shuffle
arr = np.array([1, 2, 3, 4])
np.random.shuffle(arr)
print("Shuffled:", arr)
```

## 6️⃣ Linear Algebra
- **Matrix Multiplication**: `@`, `np.dot()`, `np.matmul()`
- **Identity Matrix**: `np.eye()`
- **Inverse**: `np.linalg.inv()`
- **Determinant**: `np.linalg.det()`
- **Eigenvalues/Eigenvectors**: `np.linalg.eig()`
- **Linear Systems**: `np.linalg.solve()`
- **SVD**: `np.linalg.svd()`

**Code Example**:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
print("Matrix mul:\n", A @ B)

# Inverse
print("Inverse:\n", np.linalg.inv(A))

# Solve Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("Solution:", x)
```

## 7️⃣ Statistics and Probability
- **Central Tendency**: `np.mean()`, `np.median()`
- **Dispersion**: `np.std()`, `np.var()`
- **Correlation/Covariance**: `np.corrcoef()`, `np.cov()`
- **Percentiles**: `np.percentile()`, `np.quantile()`

**Code Example**:
```python
data = np.array([1, 2, 3, 4, 5])
print("Mean:", np.mean(data))
print("Std:", np.std(data))
print("50th percentile:", np.percentile(data, 50))

# Correlation
data2 = np.array([2, 4, 6, 8, 10])
print("Correlation:\n", np.corrcoef(data, data2))
```

## 8️⃣ Masking and Filtering
- **Boolean Masks**: Select elements with conditions
- **Conditional Selection**: `np.where()`, `np.nonzero()`

**Code Example**:
```python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
print("Mask:", mask)
print("Filtered:", arr[mask])

# np.where
print("Where > 3:", np.where(arr > 3, arr, 0))
```

## 9️⃣ Handling Missing Values
- **Missing Values**: `np.nan`
- **Functions**: `np.isnan()`, `np.nanmean()`, `np.nansum()`

**Code Example**:
```python
arr = np.array([1, np.nan, 3, 4])
print("Is NaN:", np.isnan(arr))
print("Nanmean:", np.nanmean(arr))
```

## 🔟 Data Types and Conversions
- **Dtypes**: Specify data types (e.g., `int32`, `float64`)
- **Casting**: `astype()`
- **Structured Arrays**: Arrays with named fields

**Code Example**:
```python
arr = np.array([1.5, 2.7, 3.2])
print("Float to int:", arr.astype(np.int32))

# Structured array
dt = np.dtype([('name', 'U10'), ('age', 'i4')])
structured = np.array([('Alice', 25), ('Bob', 30)], dtype=dt)
print("Structured:", structured)
```

## 1️⃣1️⃣ Memory and Performance
- **Views vs. Copies**: Views share memory; copies don’t
- **Memory Layout**: C-order, F-order
- **Broadcasting Tricks**: Avoid unnecessary copies
- **Vectorization**: Replace loops with array operations

**Code Example**:
```python
arr = np.array([1, 2, 3])
view = arr[1:]  # Shares memory
copy = arr.copy()  # New memory
print("View modified:", view.base is arr)  # True

# Vectorized operation
print("Vectorized:", arr * 2)  # Faster than loop
```

These notes summarize NumPy’s core functionality with concise examples. For deeper exploration, refer to the [NumPy documentation](https://numpy.org/doc/stable/).
