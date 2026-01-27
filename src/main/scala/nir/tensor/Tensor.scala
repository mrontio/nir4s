package nir.tensor

import io.jhdf.api.Dataset
import scala.reflect.ClassTag
import java.io.PrintWriter
import java.nio._
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import io.circe.{Decoder, Json}

/** Unified tensor class with runtime shape checking.
  *
  * Stores data in a flat array with row-major (C-style) indexing.
  *
  * @param data flat array of elements
  * @param shape dimensions of the tensor
  */
class Tensor[T: ClassTag] private (
    private val data: Array[T],
    val shape: List[Int]
) {
  require(
    data.length == shape.product,
    s"Invalid shape ${shape} for tensor of size ${data.length}"
  )

  val rank: Int = shape.length
  val size: Int = data.length

  /** Access element at given indices using row-major order. */
  def apply(indices: Int*): T = {
    val idxList = indices.toList
    require(
      idxList.length == rank,
      s"Expected $rank indices but got ${idxList.length}"
    )

    // Bounds check
    idxList.zipWithIndex.foreach { case (v, i) =>
      val bound = shape(i)
      require(
        v >= 0 && v < bound,
        s"Index $v out of bounds for axis $i (size $bound)"
      )
    }

    data(flatIndex(idxList))
  }

  /** Compute flat index from multidimensional indices (row-major). */
  private def flatIndex(indices: List[Int]): Int = {
    // For shape (2, 2, 2), strides should be (4, 2, 1)
    // scanRight computes cumulative product from the right
    val strides = shape.tail.scanRight(1)(_ * _)
    indices.zip(strides).map { case (i, s) => i * s }.sum
  }

  /** Apply function to each element. */
  def map[B: ClassTag](f: T => B): Tensor[B] =
    new Tensor[B](data.map(f), shape)

  /** Reshape tensor to new dimensions. */
  def reshape(newShape: Int*): Tensor[T] = {
    val newShapeList = newShape.toList
    val newSize = newShapeList.product
    require(
      newSize == size,
      s"New shape has size $newSize but expected $size"
    )
    new Tensor[T](data, newShapeList)
  }

  /** Remove singleton dimensions (size 1) from the left. */
  def squeeze: Tensor[T] = {
    def squeezeDims(dims: List[Int]): List[Int] = dims match {
      case Nil         => Nil
      case h :: Nil    => h :: Nil // keep last dimension
      case 1 :: t      => squeezeDims(t)
      case h :: t      => h :: squeezeDims(t)
    }
    new Tensor[T](data, squeezeDims(shape))
  }

  /** Convert to nested List structure. */
  def toList: List[_] = {
    def buildNested(offset: Int, dims: List[Int]): Any = dims match {
      case Nil => throw new IllegalStateException("Empty dimensions")
      case d :: Nil =>
        data.slice(offset, offset + d).toList
      case d :: rest =>
        val subSize = rest.product
        (0 until d).toList.map(i => buildNested(offset + i * subSize, rest))
    }
    buildNested(0, shape).asInstanceOf[List[_]]
  }

  /** Save tensor to text file for Python interop. */
  def save(path: String): Unit = {
    val writer = new PrintWriter(path)
    writer.println(shape.mkString(" "))
    data.foreach(writer.println)
    writer.close()
    println(
      s"Saved to $path. Load in Python with:\n" +
        "\timport numpy as np\n" +
        s"\twith open(\"$path\") as f:\n" +
        "\t\tshape = list(map(int, f.readline().split()))\n" +
        "\t\tdata = np.loadtxt(f)\n" +
        "\ta = data.reshape(shape)\n"
    )
  }

  override def toString: String =
    s"Tensor(shape=${shape.mkString("(", ", ", ")")})"
}

object Tensor {

  /** Create 1D tensor from varargs. */
  def apply[T: ClassTag](elements: T*): Tensor[T] =
    new Tensor[T](elements.toArray, List(elements.length))

  /** Create tensor from flat array and shape. */
  def apply[T: ClassTag](data: Array[T], shape: List[Int]): Tensor[T] = {
    val expectedSize = shape.product
    require(
      data.length == expectedSize,
      s"Array size ${data.length} doesn't match shape size $expectedSize"
    )
    new Tensor[T](data, shape)
  }

  /** Create tensor from nested list and shape. */
  def apply[T: ClassTag](list: List[_], shape: List[Int]): Tensor[T] = {
    val flatData = flattenList[T](list).toArray
    new Tensor[T](flatData, shape)
  }

  /** Create tensor from HDF5 Dataset. */
  def apply[T: ClassTag](dataset: Dataset): Tensor[T] = {
    val expectedType = implicitly[ClassTag[T]].runtimeClass
    val actualType = dataset.getDataType.getJavaType
    val shape = dataset.getDimensions.toList

    require(
      expectedType != classOf[Nothing],
      "Tensor(Dataset) constructor requires type parameter [T]"
    )
    require(
      actualType == expectedType,
      s"For dataset <${dataset.getName}>, expected type $expectedType but got $actualType"
    )

    val flatData = flattenDatasetArray[T](dataset)
    new Tensor[T](flatData, shape)
  }

  /** Load tensor from NumPy .npy file. */
  def fromNumpy[T: ClassTag](path: String): Tensor[T] = {
    val bytes = Files.readAllBytes(Paths.get(path))
    val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

    // Check magic string
    val magic = new Array[Byte](6)
    bb.get(magic)
    val isValidNumpy = magic(0) == 0x93.toByte &&
      magic(1) == 'N' &&
      magic(2) == 'U' &&
      magic(3) == 'M' &&
      magic(4) == 'P' &&
      magic(5) == 'Y'
    require(
      isValidNumpy,
      s"Not a valid .npy file (found magic bytes: ${magic.map(b => f"0x${b & 0xff}%02x").mkString(", ")})"
    )

    // Header info
    val major = bb.get()
    val minor = bb.get()
    val headerLen = bb.getShort() & 0xffff

    val headerBytes = new Array[Byte](headerLen)
    bb.get(headerBytes)
    val header = new String(headerBytes, StandardCharsets.US_ASCII)

    // Parse dtype and shape from header
    val dtype = "'descr':\\s*'([^']+)'".r
      .findFirstMatchIn(header)
      .map(_.group(1))
      .getOrElse("")
    val shape = "\\(([^)]*)\\)".r
      .findFirstMatchIn(header)
      .map(
        _.group(1).split(",").map(_.trim).filter(_.nonEmpty).map(_.toInt).toList
      )
      .getOrElse(Nil)

    val fortranOrder = "'fortran_order':\\s*(True|False)".r
      .findFirstMatchIn(header)
      .exists(_.group(1) == "True")

    require(!fortranOrder, "fromNumpy currently supports only C-order arrays")

    val dataBytes = new Array[Byte](bb.remaining())
    bb.get(dataBytes)
    val dbuf = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)

    val data: Array[Double] = dtype match {
      case "<f8" | "<d" =>
        val arr = new Array[Double](dataBytes.length / 8)
        dbuf.asDoubleBuffer().get(arr)
        arr
      case "<f4" =>
        val arr = new Array[Float](dataBytes.length / 4)
        dbuf.asFloatBuffer().get(arr)
        arr.map(_.toDouble)
      case "<i4" =>
        val arr = new Array[Int](dataBytes.length / 4)
        dbuf.asIntBuffer().get(arr)
        arr.map(_.toDouble)
      case "<i2" =>
        val arr = new Array[Short](dataBytes.length / 2)
        dbuf.asShortBuffer().get(arr)
        arr.map(_.toDouble)
      case "<u1" =>
        dataBytes.map(b => (b & 0xff).toDouble)
      case other =>
        throw new IllegalArgumentException(s"Unsupported dtype: $other")
    }

    new Tensor(data.asInstanceOf[Array[T]], shape)
  }

  // JSON decoder for Tensor[Double]
  implicit val doubleDecoder: Decoder[Tensor[Double]] = Decoder.instance {
    cursor =>
      val rawJson: Json = cursor.value

      def flatten(json: Json): List[Double] = {
        json.asArray match {
          case Some(arr) =>
            arr.toList.flatMap(flatten)
          case None =>
            json.asNumber.map(_.toDouble) match {
              case Some(d) => List(d)
              case None =>
                val errorMsg =
                  if (json.isObject)
                    s"Object with keys: ${json.asObject.map(_.keys.mkString(", ")).getOrElse("none")}"
                  else if (json.isString)
                    s"String: ${json.asString.getOrElse("")}"
                  else if (json.isBoolean)
                    s"Boolean: ${json.asBoolean.getOrElse("")}"
                  else if (json.isNull) "Null"
                  else "Unknown JSON type"
                throw new NoSuchElementException(s"Cannot parse as number: $errorMsg")
            }
        }
      }

      def inferShape(json: Json): List[Int] = {
        json.asArray match {
          case Some(arr) if arr.nonEmpty =>
            List(arr.size) ++ inferShape(arr.head)
          case Some(arr) => List(0)
          case None      => Nil
        }
      }

      val flatValues = flatten(rawJson).toArray
      val shape = inferShape(rawJson)
      Right(new Tensor(flatValues, shape))
  }

  // Helper methods
  private def flattenList[T: ClassTag](x: Any): List[T] = x match {
    case list: List[_] => list.flatMap(flattenList[T])
    case v             => List(v.asInstanceOf[T])
  }

  private def flattenDatasetArray[T: ClassTag](d: Dataset): Array[T] = {
    val rawData = d.getData
    val flat = flattenArrayRecursive(rawData)
    flat.map(_.asInstanceOf[T])
  }

  private def flattenArrayRecursive(x: Any): Array[Any] = x match {
    case a: Array[_] => a.flatMap(flattenArrayRecursive)
    case v           => Array(v)
  }
}
