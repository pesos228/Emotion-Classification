package com.example.projectneurotest;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.example.projectneurotest.ml.MinecraftKotleta;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import androidx.activity.result.PickVisualMediaRequest;


public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1;
    Bitmap loadedImage;
    int imageSize = 48;

    Dialog dialog;
    private ActivityResultLauncher<PickVisualMediaRequest> pickVisualLauncher;
    private ActivityResultLauncher<Intent> cameraLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        dialog = new Dialog(this);
        openGalleryActivity();
        cameraLauncherActivity();
    }

    public void openGallery(View view) {
        pickVisualLauncher.launch(new PickVisualMediaRequest.Builder().setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE).build());
    }
    private void openGalleryActivity() {
        pickVisualLauncher = registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
            if (uri != null) {
                try {
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                    loadedImage = convertToNeuroFormat(bitmap);
                    classifyImage(loadedImage);
                } catch (IOException e) {
                    openDialogWindow("error");
                }
            }
        });
    }

    private void openCameraPermissionGranted() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraLauncher.launch(intent);
    }
    public void openCamera(View view) {
        if(ContextCompat.checkSelfPermission(this, "Manifest.permission.CAMERA") == PackageManager.PERMISSION_GRANTED) {
            openCameraPermissionGranted();
        }
        else{ // нет разрешения
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCameraPermissionGranted();
            } else {
                // Разрешение не предоставлено
                if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) { // true если отклонил запрос, но не нажал "Не спрашивать снова". false, если выбрал "Не спрашивать снова"
                    permissionCameraDeniedTemporarily();
                } else {
                    permissionCameraDeniedForever();
                }
            }
        }
    }
    private void cameraLauncherActivity() {
        cameraLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                Bundle extras = result.getData().getExtras();
                Bitmap photoBitmap = (Bitmap) extras.get("data");
                loadedImage = convertToNeuroFormat(photoBitmap);
                classifyImage(loadedImage);
            }
        });
    }
    public void classifyImage(Bitmap image){
        try {
            MinecraftKotleta model = MinecraftKotleta.newInstance(this);


            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 1}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize*imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = image.getPixel(j, i);
                    float normalizedValue = (val & 0xFF) / 255.0f; // нормализуем значение пикселя от 0 до 1. 0xFF это маска, которая позволяет получить младшие 8 бит пикселя (только RGB)
                    byteBuffer.putFloat(normalizedValue);
                }
            }


            inputFeature0.loadBuffer(byteBuffer);


            MinecraftKotleta.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidence = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            float threshold = 0.5f; // Пороговое значение
            for(int i = 0; i<confidence.length; i++){
                if(confidence[i] > maxConfidence){
                    maxConfidence = confidence[i];
                    maxPos = i;
                }
            }
            String[] classes = {"anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"};
            if (maxConfidence < threshold) {
                openDialogWindow("none");
            } else {
                openDialogWindow(classes[maxPos]);
            }
            model.close();

        } catch (IOException e) {
            openDialogWindow("error");
            Toast.makeText(MainActivity.this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    public Bitmap convertToNeuroFormat(Bitmap originalBitmap) {
        return Bitmap.createScaledBitmap(originalBitmap, 48, 48, true);
    }
    private void openDialogWindow(String className) {
        int layoutId = getLayoutIdForClassName(className);
        if (layoutId != 0) {
            dialog.setContentView(layoutId);
            dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));

            Button buttonOK = dialog.findViewById(R.id.button_OK);
            buttonOK.setOnClickListener(v -> dialog.dismiss());
            dialog.show();
        }
        else{
            Toast.makeText(MainActivity.this, "Error", Toast.LENGTH_SHORT).show();
        }
    }
    private int getLayoutIdForClassName(String className) {
        switch (className) {
            case "anger":
                return R.layout.dialog_angry;
            case "disgust":
                return R.layout.dialog_disgust;
            case "fear":
                return R.layout.dialog_fear;
            case "happiness":
                return R.layout.dialog_happiness;
            case "sadness":
                return R.layout.dialog_sadness;
            case "surprise":
                return R.layout.dialog_surprise;
            case "neutral":
                return R.layout.dialog_neutral;
            case "none":
                return R.layout.dialog_none;
            case "error":
                return R.layout.dialog_error;
            default:
                return 0;
        }
    }

    private void permissionCameraDeniedTemporarily(){
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Permission to use the camera");
        builder.setMessage("Permission must be provided to use the camera. Please allow access to the camera.");
        builder.setPositiveButton("Allow", (dialog, which) -> {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
            dialog.dismiss();
        });
        builder.setNegativeButton("Cancel", (dialog, which) -> dialog.dismiss());
        AlertDialog dialog = builder.create();
        dialog.show();
    }
    private void permissionCameraDeniedForever(){
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Permission to use the camera");
        builder.setMessage("To use the camera, you need to give permission for this in the settings.");
        builder.setPositiveButton("Ok", (dialog, which) -> dialog.dismiss());
        AlertDialog dialog = builder.create();
        dialog.show();
    }
}