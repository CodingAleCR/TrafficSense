package cr.codingale.trafficsense;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

class TrafficSenseProcessor {

    public enum Salida {
        ENTRADA, INTENSIDAD, OPERADOR_LOCAL, BINARIZACION,
        SEGMENTACION, RECONOCIMIENTO
    }

    public enum TipoIntensidad {
        SIN_PROCESO,
        LUMINANCIA,
        AUMENTO_LINEAL_CONSTRASTE,
        EQUALIZ_HISTOGRAMA,
        ZONAS_ROJAS,
        ZONAS_VERDES,
    }

    public enum TipoOperadorLocal {
        SIN_PROCESO,
        PASO_BAJO,
        PASO_ALTO,
        SOBEL_ILUMINANCIA,
        SOBEL_ROJAS,
        SOBEL_VERDES,
        MORFOLOGICO,
        RESIDUO_DILATACION,
    }

    public enum TipoBinarizacion {
        SIN_PROCESO,
        ADAPTATIVA,
        MAXIMO
    }

    public enum TipoSegmentacion {
        SIN_PROCESO
    }

    public enum TipoReconocimiento {
        SIN_PROCESO
    }

    private Mat gris;
    private Mat salidaintensidad;
    private Mat salidatrlocal;
    private Mat salidabinarizacion;
    private Mat salidasegmentacion;
    private Mat salidaocr;

    private Salida mostrarSalida;
    private TipoIntensidad tipoIntensidad;
    private TipoOperadorLocal tipoOperadorLocal;
    private TipoBinarizacion tipoBinarizacion;
    private TipoSegmentacion tipoSegmentacion;
    private TipoReconocimiento tipoReconocimiento;

    TrafficSenseProcessor() {
        mostrarSalida = Salida.INTENSIDAD;
        tipoIntensidad = TipoIntensidad.LUMINANCIA;
        tipoOperadorLocal = TipoOperadorLocal.SIN_PROCESO;
        tipoBinarizacion = TipoBinarizacion.SIN_PROCESO;
        tipoSegmentacion = TipoSegmentacion.SIN_PROCESO;
        tipoReconocimiento = TipoReconocimiento.SIN_PROCESO;
        salidaintensidad = new Mat();
        salidatrlocal = new Mat();
        salidabinarizacion = new Mat();
        salidasegmentacion = new Mat();
        salidaocr = new Mat();
        gris = new Mat();
    }

    public Mat getGris() {
        return gris;
    }

    public void setGris(Mat gris) {
        this.gris = gris;
    }

    public Mat getSalidaintensidad() {
        return salidaintensidad;
    }

    public void setSalidaintensidad(Mat salidaintensidad) {
        this.salidaintensidad = salidaintensidad;
    }

    public Mat getSalidatrlocal() {
        return salidatrlocal;
    }

    public void setSalidatrlocal(Mat salidatrlocal) {
        this.salidatrlocal = salidatrlocal;
    }

    public Mat getSalidabinarizacion() {
        return salidabinarizacion;
    }

    public void setSalidabinarizacion(Mat salidabinarizacion) {
        this.salidabinarizacion = salidabinarizacion;
    }

    public Mat getSalidasegmentacion() {
        return salidasegmentacion;
    }

    public void setSalidasegmentacion(Mat salidasegmentacion) {
        this.salidasegmentacion = salidasegmentacion;
    }

    public Mat getSalidaocr() {
        return salidaocr;
    }

    public void setSalidaocr(Mat salidaocr) {
        this.salidaocr = salidaocr;
    }

    public Salida getMostrarSalida() {
        return mostrarSalida;
    }

    public void setMostrarSalida(Salida mostrarSalida) {
        this.mostrarSalida = mostrarSalida;
    }

    public TipoIntensidad getTipoIntensidad() {
        return tipoIntensidad;
    }

    public void setTipoIntensidad(TipoIntensidad tipoIntensidad) {
        this.tipoIntensidad = tipoIntensidad;
    }

    public TipoOperadorLocal getTipoOperadorLocal() {
        return tipoOperadorLocal;
    }

    public void setTipoOperadorLocal(TipoOperadorLocal tipoOperadorLocal) {
        this.tipoOperadorLocal = tipoOperadorLocal;
    }

    public TipoBinarizacion getTipoBinarizacion() {
        return tipoBinarizacion;
    }

    public void setTipoBinarizacion(TipoBinarizacion tipoBinarizacion) {
        this.tipoBinarizacion = tipoBinarizacion;
    }

    public TipoSegmentacion getTipoSegmentacion() {
        return tipoSegmentacion;
    }

    public void setTipoSegmentacion(TipoSegmentacion tipoSegmentacion) {
        this.tipoSegmentacion = tipoSegmentacion;
    }

    public TipoReconocimiento getTipoReconocimiento() {
        return tipoReconocimiento;
    }

    public void setTipoReconocimiento(TipoReconocimiento tipoReconocimiento) {
        this.tipoReconocimiento = tipoReconocimiento;
    }

    Mat digest(Mat entrada) {
        if (mostrarSalida == Salida.ENTRADA) {
            return entrada;
        }
        // Transformación intensidad
        switch (tipoIntensidad) {
            case SIN_PROCESO:
                salidaintensidad = entrada;
                break;
            case LUMINANCIA:
                Imgproc.cvtColor(entrada, salidaintensidad,
                        Imgproc.COLOR_RGBA2GRAY);
                break;
            case AUMENTO_LINEAL_CONSTRASTE:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                aumentoLinealConstraste(gris); //resultado en salida intensidad
                break;
            case EQUALIZ_HISTOGRAMA:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                //Eq. Hist necesita gris
                Imgproc.equalizeHist(gris, salidaintensidad);
                break;
            case ZONAS_ROJAS:
                zonaRoja(entrada); //resultado en salidaintensidad
                break;
            case ZONAS_VERDES:
                zonaVerde(entrada); //resultado en salidaintensidad
                break;
            default:
                salidaintensidad = entrada;
        }
        if (mostrarSalida == Salida.INTENSIDAD) {
            return salidaintensidad;
        }

        // Operador local
        switch (tipoOperadorLocal) {
            case SIN_PROCESO:
                salidatrlocal = entrada;
                break;
            case PASO_BAJO:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                pasoBajo(gris); //resultado en salidatrlocal
                break;

            case PASO_ALTO:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                pasoAlto(gris);
                break;

            case SOBEL_ILUMINANCIA:
                sobelLuminancia(entrada);
                break;

            case SOBEL_ROJAS:
                sobelRojas(entrada);
                break;

            case SOBEL_VERDES:
                sobelVerdes(entrada);
                break;

            case MORFOLOGICO:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                residuoDilatacion(gris, 3);
                break;

            case RESIDUO_DILATACION:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                residuoDilatacion(gris, 11);
                break;

            default:
                salidatrlocal = entrada;
                break;
        }
        if (mostrarSalida == Salida.OPERADOR_LOCAL) {
            return salidatrlocal;
        }

        // Binarización
        switch (tipoBinarizacion) {
            case SIN_PROCESO:
                salidabinarizacion = salidatrlocal;
                break;
            default:
                salidabinarizacion = salidatrlocal;
                break;
        }
        if (mostrarSalida == Salida.BINARIZACION) {
            return salidabinarizacion;
        }
        switch (tipoSegmentacion) {
            case SIN_PROCESO:
                salidasegmentacion = salidabinarizacion;
                break;
        }
        if (mostrarSalida == Salida.SEGMENTACION) {
            return salidasegmentacion;
        }

        // Reconocimiento OCR
        switch (tipoReconocimiento) {
            case SIN_PROCESO:
                salidaocr = salidasegmentacion;
                break;
        }
        return salidaocr;
    }

    void splitScreen(Mat entrada, Mat salida) {
        if (entrada.channels() > salida.channels())
            Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        if (entrada.channels() < salida.channels())
            Imgproc.cvtColor(entrada, entrada, Imgproc.COLOR_GRAY2RGBA);
        //Representar la entrada en la mitad izquierda
        Rect mitad_izquierda = new Rect();
        mitad_izquierda.x = 0;
        mitad_izquierda.y = 0;
        mitad_izquierda.height = entrada.height();
        mitad_izquierda.width = entrada.width() / 2;
        Mat salida_mitad_izquierda = salida.submat(mitad_izquierda);
        Mat entrada_mitad_izquierda = entrada.submat(mitad_izquierda);
        entrada_mitad_izquierda.copyTo(salida_mitad_izquierda);
    }

    private void zonaRoja(Mat entrada) { //Ejemplo para ser rellenado en curso
        Mat red = new Mat();
        Mat green = new Mat();
        Mat blue = new Mat();
        Mat maxGB = new Mat();

        salidaintensidad = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract(red, maxGB, salidaintensidad);
    }

    private void zonaVerde(Mat entrada) { //Ejemplo para ser rellenado en curso
        Mat red = new Mat();
        Mat green = new Mat();
        Mat blue = new Mat();
        Mat maxRB = new Mat();

        salidaintensidad = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(red, blue, maxRB);
        Core.subtract(green, maxRB, salidaintensidad);
    }

    private void aumentoLinealConstraste(Mat entrada) { //Ejemplo para ser rellenado
        MatOfInt canales = new MatOfInt(0);
        MatOfInt numero_bins = new MatOfInt(256);
        MatOfFloat intervalo = new MatOfFloat(0, 256);
        Mat hist = new Mat();
        List<Mat> imagenes = new ArrayList<Mat>();
        float[] histograma = new float[256];


        salidaintensidad = new Mat();
        imagenes.clear(); //Eliminar imagen anterior si la hay
        imagenes.add(entrada); //Añadir imagen actual
        Imgproc.calcHist(imagenes, canales, new Mat(), hist,
                numero_bins, intervalo);

        //Lectura del histograma a un array de float
        hist.get(0, 0, histograma);

        //Calcular xmin y xmax
        int total_pixeles = entrada.cols() * entrada.rows();
        float porcentaje_saturacion = (float) 0.05;
        int pixeles_saturados = (int) (porcentaje_saturacion
                * total_pixeles);
        int xmin = 0;
        int xmax = 255;
        float acumulado = 0f;
        for (int n = 0; n < 256; n++) { //xmin
            acumulado = acumulado + histograma[n];
            if (acumulado > pixeles_saturados) {
                xmin = n;
                break;
            }
        }
        acumulado = 0;
        for (int n = 255; n >= 0; n--) { //xmax
            acumulado = acumulado + histograma[n];
            if (acumulado > pixeles_saturados) {
                xmax = n;
                break;
            }
        }

        //Calculo de la salida
        Core.subtract(entrada, new Scalar(xmin), salidaintensidad);
        float pendiente = ((float) 255.0) / ((float) (xmax - xmin));
        Core.multiply(salidaintensidad, new Scalar(pendiente), salidaintensidad);
    }

    private void pasoBajo(Mat entrada) { //Ejemplo para ser rellenado
        Mat paso_bajo = new Mat();
        int filter_size = 17;
        Size s = new Size(filter_size, filter_size);
        Imgproc.blur(entrada, paso_bajo, s);

        salidatrlocal = paso_bajo;
    }

    private void pasoAlto(Mat entrada) { //Ejemplo para ser rellenado
        Mat salida = new Mat();

        Mat paso_bajo = new Mat();
        int filter_size = 17;
        Size s = new Size(filter_size, filter_size);
        Imgproc.blur(entrada, paso_bajo, s);

        // Hacer la resta. Los valores negativos saturan a cero
        Core.subtract(paso_bajo, entrada, salida);

        //Aplicar Ganancia para ver mejor. La multiplicacion satura
        Scalar ganancia = new Scalar(2);
        Core.multiply(salida, ganancia, salida);

        salidatrlocal = salida;
    }

    private void modGradiente(Mat entrada) {
        Mat Gx = new Mat();
        Mat Gy = new Mat();
        Imgproc.Sobel(entrada, Gx, CvType.CV_32FC1, 1, 0);
        //Derivada primera rto x
        Imgproc.Sobel(entrada, Gy, CvType.CV_32FC1, 0, 1);
        //Derivada primera rto y

        Mat Gx2 = new Mat();
        Mat Gy2 = new Mat();
        Core.multiply(Gx, Gx , Gx2); //Gx2 = Gx*Gx elemento a elemento
        Core.multiply(Gy, Gy , Gy2); //Gy2 = Gy*Gy elemento a elemento
        Mat ModGrad2 = new Mat();
        Core.add( Gx2 , Gy2, ModGrad2);
        Mat ModGrad = new Mat();
        Core.sqrt(ModGrad2,ModGrad);

        ModGrad.convertTo(salidatrlocal, CvType.CV_8UC1);
    }

    private void sobelLuminancia(Mat entrada) { //Ejemplo para ser rellenado
        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
        modGradiente(gris);
    }

    private void sobelRojas(Mat entrada) { //Ejemplo para ser rellenado
        Mat cRojas = new Mat();
        Core.extractChannel(entrada, cRojas, 0);
        modGradiente(cRojas);
    }

    private void sobelVerdes(Mat entrada) { //Ejemplo para ser rellenado
        Mat cRojas = new Mat();
        Core.extractChannel(entrada, cRojas, 1);
        modGradiente(cRojas);
    }

    private void residuoDilatacion(Mat entrada, double tam) { //Ejemplo para ser rellenado
        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
                Size(tam,tam));
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(entrada, gray_dilation, SE ); // 3x3 dilation
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, entrada, dilation_residue);

        salidatrlocal = dilation_residue;
    }
}


