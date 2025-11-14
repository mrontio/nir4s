package nir.tensor

import io.jhdf.api.Dataset
import scala.reflect.ClassTag
import scala.util.matching.Regex
import scala.language.implicitConversions
import java.io.PrintWriter
import java.nio.file.Paths
import java.nio._
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.reflect.ClassTag


// JSON Parsing
import io.circe.{Decoder, HCursor, Json}

/* Dynamic Tensor
 * Dynamic construction of a tensor from runtime / unknown shape size.
 * Advantages: dynamic construction, reshaping, 1D backing array.
 * Disadvantages: Slower runtime due to checks, often requires asInstanceOf
 */


/** Utility to translate multidimensional tensor indices into a single
  * row-major offset.
  *
  * For a tensor of shape `(2, 3)` the index `(1, 2)` is computed as
  * \(1 \times 3 + 2 = 5\).
  *
  * @param shape dimensions of the tensor in each axis
  */
case class Indexer(shape: List[Int]) {

  def rank: Int = shape.length
  def size: Int = shape.reduce(_ * _)
  def rangeTree: RangeTree = RangeTree.buildTree(shape)

  private def idx(dims: List[Int]): Int = {
    // Row-major order (like numpy), (0, 0, 1) -> 1
    val shapeCumprod: List[Int] = shape.reverse.scanLeft(1)(_ * _).tail
    dims match {
      case Nil        => 0
      case dim :: Nil => dim
      case dim :: t   => idx(t) + dim * shapeCumprod(t.length - 1)
    }
  }

  def apply(vals: Seq[Int]): Int = {
    val vs = vals.toList
    // Rank check
    if (vs.length != rank) throw new IllegalArgumentException(s"Expected ${rank} indices but got ${vs.length}")

    // Bound check
    vs.zipWithIndex.foreach { case (v, i) =>
      val bound = shape(i)
      if (!(v >= 0 && v < bound)) throw new IndexOutOfBoundsException(s"Index $v out of bounds for axis $i (size $bound)")
    }

    // Run indexer
    idx(vs)
  }

  private def squeeze(shape: List[Int]): List[Int] =
    shape match {
      case Nil => Nil
      // case 1 :: Nil => Nil -- we assume we don't want to squeeze from the right
      case h :: Nil => h :: Nil
      case 1 :: t => squeeze(t)
      case h :: t => h :: squeeze(t)
    }

  def squeeze(): Indexer = {
    val ss = squeeze(shape)
    Indexer(ss)
  }

  def reshape(newshape: List[Int]): Indexer = {
    val newsize = newshape.reduce(_ * _)
    if (newsize != size) throw new IllegalArgumentException(s"New shape has size $newsize but expected $size.")

    Indexer(newshape)
  }
}

class TensorDynamic[D: ClassTag](data: Array[D], idx: Indexer) {
  assert(data.length == idx.shape.reduce(_ * _), s"Invalid shape ${idx.shape} for tensor of size ${data.length}")

  val shape = idx.shape
  val rank = idx.rank
  val size = idx.size
  def apply(indices: Int*) = data(idx(indices))
  def reshape(newshape: Int*): TensorDynamic[D] = new TensorDynamic[D](data, idx.reshape(newshape.toList))
  def squeeze: TensorDynamic[D] = new TensorDynamic(data, idx.squeeze())
  def map[B: ClassTag](f: D => B): TensorDynamic[B] = new TensorDynamic[B](data.map(f), idx)

  private def toNestedList(tree: RangeTree): List[Any] = tree match {
    case Leaf(b, e) =>
      data.slice(b, e).toList

    case Branch(children) =>
      children.map(toNestedList)
  }

  def toList: List[_] = toNestedList(idx.rangeTree)

  // Map of Dynamic.Rank -> Static constructors
  private val staticConstructors: Map[Int, (Array[D], RangeTree) => TensorStatic[D]] = Map(
    1 -> Tensor1D.fromRangeTree[D],
    2 -> Tensor2D.fromRangeTree[D],
    3 -> Tensor3D.fromRangeTree[D],
    4 -> Tensor4D.fromRangeTree[D],
  )

  def toStatic: TensorStatic[D] = {
    staticConstructors.get(rank) match {
      case Some(constructor) => constructor(data, idx.rangeTree)
      case None => throw new Exception(s"Static tensors of dimension $rank are not yet supported.")
    }
  }

  def save(path: String): Unit = {
    val writer = new PrintWriter(s"$path")

    writer.println(shape.mkString(" "))
    data.foreach(writer.println)
    writer.close()
    println(s"Save to ${path} success, use this file from Python with:\n" +
      "\timport numpy as np\n" +
      s"\twith open(\"$path\") as f:\n" +
      "\t\tshape = list(map(int, f.readline().split()))\n" +
      "\t\tdata = np.loadtxt(f)\n" +
      "\ta = data.reshape(shape)\n")
  }


}

object TensorDynamic {
  // Allow to match based on rank
  object Rank {
    def unapply[D](tensor: TensorDynamic[D]): Option[Int] = Some(tensor.rank)
  }

  def flattenArray[T: ClassTag](a: Any): Any = {
    if (!a.isInstanceOf[Array[_]]) {
      throw new IllegalArgumentException("Input must be an array")
    }
    val flattened = flattenArrayRecursive(a).toArray
    flattened
  }

  private def flattenArrayRecursive(x: Any): Array[Any] = x match {
    case a: Array[_] => a.flatMap(flattenArrayRecursive)
    case v           => Array(v)
  }

  def flattenList[T: ClassTag](a: Any): List[T] = {
    if (!a.isInstanceOf[List[_]]) {
      throw new IllegalArgumentException("Input must be List[_]")
    }
    val flattened = flattenListRecursive(a).toList.asInstanceOf[List[T]]
    flattened
  }

  private def flattenListRecursive(x: Any): List[Any] = x match {
    case a: List[_] => a.flatMap(flattenListRecursive)
    case v           => List(v)
  }


  private def flattenDatasetArray[T: ClassTag](d: Dataset): Array[T] = {
    val rawData = d.getData // e.g. Array[Float], Array[Array[Float]], etc.
    val flat: Array[Any] = flattenArrayRecursive(rawData)
    flat.map(_.asInstanceOf[T]) // safe because caller controls T
  }

  // 1D case
  def apply[D: ClassTag](a: D*): TensorDynamic[D] = {
    val i = Indexer(List(a.length))
    new TensorDynamic[D](a.toArray, i)
  }

  // N-D case.
  def apply[D: ClassTag](a: Array[D], shape: List[Int]): TensorDynamic[D] = {
    val i = Indexer(shape)
    if (i.size != a.length) throw new IllegalArgumentException(s"Supplied array is size ${a.length} but shape is over size ${i.size}.")
    new TensorDynamic[D](a, i)
  }

  def apply[D: ClassTag](d: Dataset): TensorDynamic[D] = {
    val expectedType = implicitly[ClassTag[D]].runtimeClass
    val actualType = d.getDataType.getJavaType
    val idx = Indexer(d.getDimensions.toList)

    // Run-time checks
    require(expectedType != classOf[Nothing], s"TensorDynamic(Dataset) contructor requires type parameter [D].")
    require(actualType == expectedType, s"For dataset <${d.getName}>, expected type $expectedType but got $actualType.")

    new TensorDynamic(flattenDatasetArray[D](d), idx)
  }

  def apply[D: ClassTag](l: List[_], shape: List[Int]): TensorDynamic[D] = {
    val idx = Indexer(shape)
    val data: Array[D] = flattenList(l).asInstanceOf[List[D]].toArray
    new TensorDynamic(data, idx)
  }

  // This is ChatGPT-d due to time, need to look over later.
  def fromNumpy[D: ClassTag](path: String): TensorDynamic[D] = {
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
    require(isValidNumpy, s"Not a valid .npy file (found magic bytes: ${magic.map(b => f"0x${b & 0xFF}%02x").mkString(", ")})")


    // Header info
    val major = bb.get()
    val minor = bb.get()
    val headerLen = bb.getShort() & 0xFFFF

    val headerBytes = new Array[Byte](headerLen)
    bb.get(headerBytes)
    val header = new String(headerBytes, StandardCharsets.US_ASCII)

    // Parse dtype and shape from header
    val dtype = "'descr':\\s*'([^']+)'".r.findFirstMatchIn(header).map(_.group(1)).getOrElse("")
    val shape = "\\(([^)]*)\\)".r.findFirstMatchIn(header)
      .map(_.group(1).split(",").map(_.trim).filter(_.nonEmpty).map(_.toInt).toList)
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
        dataBytes.map(b => (b & 0xFF).toDouble)
      case other =>
        throw new IllegalArgumentException(s"Unsupported dtype: $other")
    }

    val idx = Indexer(shape)
    new TensorDynamic(data.asInstanceOf[Array[D]], idx)
  }

  // Implicit conversions to Static
  implicit def toStaticConv[T](td: TensorDynamic[T]): TensorStatic[T] = td.toStatic


  implicit val doubleLoader: Decoder[TensorDynamic[Double]] = Decoder.instance { cursor =>
    val rawJson: Json = cursor.value
    def flatten(json: Json): List[Double] = {
      json.asArray match {
        case Some(arr) =>
          arr.toList.flatMap(flatten)
        case None =>
          json.asNumber.map(_.toDouble) match {
            case Some(d) =>
              List(d)
            case None =>
              val errorMsg = if (json.isObject) {
                s"❌ Object with keys: ${json.asObject.map(_.keys.mkString(", ")).getOrElse("none")}"
              } else if (json.isString) {
                s"❌ String: ${json.asString.getOrElse("")}"  // Other type to consider
              } else if (json.isBoolean) {
                s"❌ Boolean: ${json.asBoolean.getOrElse("")}"  // Other type to consider
              } else if (json.isNull) {
                s"❌ Null"  // Other type to consider
              } else {
                s"❌ Unknown JSON type"
              }
              println(errorMsg)
              throw new NoSuchElementException(errorMsg)
          }
      }
    }

    def inferShape(json: Json): List[Int] = {
      json.asArray match {
        case Some(arr) if arr.nonEmpty =>
          val headShape = inferShape(arr.head)
          List(arr.size) ++ headShape
        case Some(arr) => List(0)
        case None => Nil
      }
    }

    val flatValues = flatten(rawJson).toArray
    val indexer = new Indexer(inferShape(rawJson))
    Right(new TensorDynamic(flatValues, indexer))
  }
}
