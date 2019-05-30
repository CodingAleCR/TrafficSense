package cr.codingale.trafficsense;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "OCVSample::MainActivity";

    private CameraBridgeViewBase mCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == BaseLoaderCallback.SUCCESS) {
                mCameraView.setMaxFrameSize(mCamWidth, mCamHeight);
                mCameraView.enableView();
            } else {
                Log.e(TAG, "OpenCV no se cargo");
                Toast.makeText(MainActivity.this, "OpenCV no se cargo",
                        Toast.LENGTH_LONG).show();
                finish();
            }
        }
    };

    private int mCamIndex; // 0-> camara trasera; 1-> camara frontal
    private int mCamWidth = 320;// resolucion deseada de la imagen
    private int mCamHeight = 240;
    private static final String STATE_CAMERA_INDEX = "cameraIndex";

    private int mInputType = 0; // 0 -> cámara 1 -> fichero1 2 -> fichero2
    private Mat mResourceImage_;
    private boolean mShouldReloadResource = false;
    private boolean mShouldSaveNextImage = false;

    private TrafficSenseProcessor mProcessor;

    private boolean mSplitScreen = false;

    private int startGC = 15;

    public void onSaveInstanceState(Bundle savedInstanceState) {
// Save the current camera index.
        savedInstanceState.putInt(STATE_CAMERA_INDEX, mCamIndex);
        super.onSaveInstanceState(savedInstanceState);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new
                        String[]{Manifest.permission.CAMERA}, 1);
            }
        }

        mCameraView = findViewById(R.id.java_surface_view);
        mCameraView.setVisibility(SurfaceView.VISIBLE);
        mCameraView.setCvCameraViewListener(this);

        if (savedInstanceState != null) {
            mCamIndex = savedInstanceState.getInt(STATE_CAMERA_INDEX, 0);
        } else {
            mCamIndex = CameraBridgeViewBase.CAMERA_ID_BACK;
        }
        mCameraView.setCameraIndex(mCamIndex);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        super.onCreateOptionsMenu(menu);
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.cambiarCamara:
                if (mCamIndex == CameraBridgeViewBase.CAMERA_ID_BACK) {
                    mCamIndex = CameraBridgeViewBase.CAMERA_ID_FRONT;
                } else mCamIndex = CameraBridgeViewBase.CAMERA_ID_BACK;
                recreate();
                break;
            case R.id.resolucion_800x600:
                mCamWidth = 800;
                mCamHeight = 600;
                restartResolution();
                break;
            case R.id.resolucion_640x480:
                mCamWidth = 640;
                mCamHeight = 480;
                restartResolution();
                break;
            case R.id.resolucion_320x240:
                mCamWidth = 320;
                mCamHeight = 240;
                restartResolution();
                break;

            case R.id.entrada_camara:
                mInputType = 0;
                break;
            case R.id.entrada_fichero1:
                mInputType = 1;
                mShouldReloadResource = true;
                break;
            case R.id.entrada_fichero2:
                mInputType = 2;
                mShouldReloadResource = true;
                break;
            case R.id.guardar_imagenes:
                mShouldSaveNextImage = true;
                break;
            case R.id.preferencias:
                Intent i = new Intent(this, SettingsActivity.class);
                startActivity(i);
                break;
        }
        String msg = "W=" + mCamWidth + " H= " +
                mCamHeight + " Cam= " +
                Integer.toBinaryString(mCamIndex);
        Toast.makeText(MainActivity.this, msg,
                Toast.LENGTH_LONG).show();
        return true;
    }

    public boolean onTouchEvent(MotionEvent event) {
        openOptionsMenu();
        return true;
    }

    @Override
    public void openOptionsMenu() {
        super.openOptionsMenu();
        Configuration config = getResources().getConfiguration();
        if ((config.screenLayout & Configuration.SCREENLAYOUT_SIZE_MASK) > Configuration.SCREENLAYOUT_SIZE_LARGE) {
            int originalScreenLayout = config.screenLayout;
            config.screenLayout = Configuration.SCREENLAYOUT_SIZE_LARGE;
            super.openOptionsMenu();
            config.screenLayout = originalScreenLayout;
        } else {
            super.openOptionsMenu();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mCameraView != null) {
            mCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mCameraView != null) {
            mCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mCamWidth = width;
        mCamHeight = height;

        mProcessor = new TrafficSenseProcessor();

        PreferenceManager.setDefaultValues(this, R.xml.preferencias, false);
        SharedPreferences preferencias = PreferenceManager
                .getDefaultSharedPreferences(this);
        mSplitScreen = (preferencias.getBoolean("pantalla_partida", true));
        String valor = preferencias.getString("salida", "ENTRADA");
        mProcessor.setMostrarSalida(TrafficSenseProcessor.Salida.valueOf(valor));
        valor = preferencias.getString("intensidad", "SIN_PROCESO");
        mProcessor.setTipoIntensidad(TrafficSenseProcessor.TipoIntensidad.valueOf(valor));
        valor = preferencias.getString("operador_local", "SIN_PROCESO");
        mProcessor.setTipoOperadorLocal(TrafficSenseProcessor.TipoOperadorLocal.valueOf(valor));
        valor = preferencias.getString("binarizacion", "SIN_PROCESO");
        mProcessor.setTipoBinarizacion(TrafficSenseProcessor.TipoBinarizacion.valueOf(valor));
        valor = preferencias.getString("segmentacion", "SIN_PROCESO");
        mProcessor.setTipoSegmentacion(TrafficSenseProcessor.TipoSegmentacion.valueOf(valor));
        valor = preferencias.getString("reconocimiento", "SIN_PROCESO");
        mProcessor.setTipoReconocimiento(TrafficSenseProcessor.TipoReconocimiento.valueOf(valor));
    }

    @Override
    public void onCameraViewStopped() {
        //Not used.
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat input;
        if (mInputType == 0) {
            input = inputFrame.rgba();
        } else {
            if (mShouldReloadResource) {
                mResourceImage_ = new Mat();
                //Poner aqui el nombre de los archivos copiados
                int[] RECURSOS_FICHEROS = {0, R.raw.img1, R.raw.img2};
                Bitmap bitmap = BitmapFactory.decodeResource(getResources(),
                        RECURSOS_FICHEROS[mInputType]);
                //Convierte el recurso a una Mat de OpenCV
                Utils.bitmapToMat(bitmap, mResourceImage_);
                Imgproc.resize(mResourceImage_, mResourceImage_,
                        new Size(mCamWidth, mCamHeight));
                mShouldReloadResource = false;
            }
            input = mResourceImage_;
        }
        Mat output = mProcessor.digest(input);
        if (mSplitScreen) {
            mProcessor.splitScreen(input, output);
        }
        if (mShouldSaveNextImage) { //Para foto salida debe ser rgba
            takePhoto(input, output);
            mShouldSaveNextImage = false;
        }
        if (mInputType > 0) {
            //Es necesario que el tamaño de la salida coincida con el real de captura
            Imgproc.resize(output, output, new Size(mCamWidth, mCamHeight));
        }

        //Memory release
        startGC--;
        if (startGC == 0) {
            System.gc();
            System.runFinalization();
            startGC = 15;
        }


        return output;
    }

    private void takePhoto(final Mat input, final Mat output) {
        // Determina la ruta para crear los archivos
        final long currentTimeMillis = System.currentTimeMillis();
        final String appName = getString(R.string.app_name);
        final String galleryPath = Environment
                .getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_PICTURES).toString();
        final String albumPath = galleryPath + "/" + appName;
        final String photoPathIn = albumPath + "/In_" + currentTimeMillis
                + ".png";
        final String photoPathOut = albumPath + "/Out_" + currentTimeMillis
                + ".png";

        // Asegurarse que el directorio existe
        File album = new File(albumPath);
        if (!album.isDirectory() && !album.mkdirs()) {
            Log.e(TAG, "Error al crear el directorio " + albumPath);
            return;
        }

        // Intenta crear los archivos
        Mat mBgr = new Mat();
        if (output.channels() == 1)
            Imgproc.cvtColor(output, mBgr, Imgproc.COLOR_GRAY2BGR, 3);
        else
            Imgproc.cvtColor(output, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
        if (!Imgcodecs.imwrite(photoPathOut, mBgr)) {
            Log.e(TAG, "Fallo al guardar " + photoPathOut);
        }
        if (input.channels() == 1)
            Imgproc.cvtColor(input, mBgr, Imgproc.COLOR_GRAY2BGR, 3);
        else
            Imgproc.cvtColor(input, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
        if (!Imgcodecs.imwrite(photoPathIn, mBgr))
            Log.e(TAG, "Fallo al guardar " + photoPathIn);
        mBgr.release();
    }

    public void restartResolution() {
        mCameraView.disableView();
        mCameraView.setMaxFrameSize(mCamWidth, mCamHeight);
        mCameraView.enableView();
    }
}
