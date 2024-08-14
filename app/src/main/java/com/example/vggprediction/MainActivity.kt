package com.example.vggprediction

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.vggprediction.ml.TfliteModelVgg16
import com.example.vggprediction.ml.TfliteModelVgg19
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var selectBtn: Button
    private lateinit var predictBtn: Button
    private lateinit var resultView: TextView
    private lateinit var imageView: ImageView
    private var bitmap: Bitmap? = null
    private lateinit var progressBar: ProgressBar
    private lateinit var imageProcessor: ImageProcessor

    private lateinit var modelVGG16: TfliteModelVgg16
    private lateinit var modelVGG19: TfliteModelVgg19

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.selectBtn)
        predictBtn = findViewById(R.id.predictBtn)
        resultView = findViewById(R.id.resultView)
        imageView = findViewById(R.id.imageView)
        progressBar = findViewById(R.id.progressBar)

        // Inisialisasi ImageProcessor
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        selectBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

        modelVGG16 = TfliteModelVgg16.newInstance(this)
        modelVGG19 = TfliteModelVgg19.newInstance(this)

        predictBtn.setOnClickListener {
            if (bitmap != null) {
                resultView.text = "Loading..."

                lifecycleScope.launch(Dispatchers.Default) {
                    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)

                    // Preprocess gambar
                    val tensorImage = TensorImage(DataType.FLOAT32)
                    tensorImage.load(bitmap!!)
                    val processedImage = imageProcessor.process(tensorImage)
                    inputFeature0.loadBuffer(processedImage.buffer)

                    // VGG19
                    val outputsVGG19 = modelVGG19.process(inputFeature0)
                    val outputFeature0VGG19 = outputsVGG19.outputFeature0AsTensorBuffer.floatArray
                    val glassesProbabilityVGG19 = outputFeature0VGG19[0]

                    // VGG16
                    val outputsVGG16 = modelVGG16.process(inputFeature0)
                    val outputFeature0VGG16 = outputsVGG16.outputFeature0AsTensorBuffer.floatArray
                    val glassesProbabilityVGG16 = outputFeature0VGG16[0]

                    withContext(Dispatchers.Main) {

                        val resultTextVGG19 = if (glassesProbabilityVGG19 > 0.5) {
                            "VGG19: Kacamata terdeteksi (probabilitas: ${String.format("%.2f", glassesProbabilityVGG19 * 100)}%)"
                        } else {
                            "VGG19: Tidak ada kacamata (probabilitas: ${String.format("%.2f", (1 - glassesProbabilityVGG19) * 100)}%)"
                        }

                        val resultTextVGG16 = if (glassesProbabilityVGG16 > 0.5) {
                            "VGG16: Kacamata terdeteksi (probabilitas: ${String.format("%.2f", glassesProbabilityVGG16 * 100)}%)"
                        } else {
                            "VGG16: Tidak ada kacamata (probabilitas: ${String.format("%.2f", (1 - glassesProbabilityVGG16) * 100)}%)"
                        }

                        resultView.text = "$resultTextVGG19\n$resultTextVGG16" // Tampilkan hasil
                    }
                }
            } else {
                Toast.makeText(this, "Silakan pilih gambar terlebih dahulu", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100 && resultCode == RESULT_OK) {
            val uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
            imageView.setImageBitmap(bitmap)
        }
    }
}
