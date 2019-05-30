package cr.codingale.trafficsense;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

class TrafficSenseProcessor {

    public enum Salida {
        ENTRADA,
        INTENSIDAD,
        OPERADOR_LOCAL,
        BINARIZACION,
        SEGMENTACION,
        RECONOCIMIENTO,
        CAR_MODE,
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
        ZONAS_ROJAS,
        ADAPTATIVA,
        OTSU
    }

    public enum TipoSegmentacion {
        SIN_PROCESO,
        SELECCION_CANDIDATOS,
        ANIDADAS,
        ZONAS_ROJAS,
        DIGITOS_SENAL,
    }

    public enum TipoReconocimiento {
        SIN_PROCESO,
        OCR1,
        LTROCR,
    }

    private Mat gris;
    private Mat salidaintensidad;
    private Mat salidatrlocal;
    private Mat salidabinarizacion;
    private Mat salidasegmentacion;
    private Mat salidaocr;
    private Mat salidaCarro;

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
        tabla_caracteristicas = new Mat(NUMERO_CLASES * MUESTRAS_POR_CLASE,
                NUMERO_CARACTERISTICAS, CvType.CV_64FC1);
        binaria1 = new Mat();
        binaria2 = new Mat();
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
                salidabinarizacion = entrada;
                break;
            case ZONAS_ROJAS:
                binarizacionRojas(entrada);
                break;
            case ADAPTATIVA:
                binarizacionAdaptativa(entrada);
                break;
            case OTSU:
                binarizacionOtsu(entrada);
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
                salidasegmentacion = entrada;
                break;

            case SELECCION_CANDIDATOS:
                seleccionCandidatosCirculos(entrada);
                break;

            case ANIDADAS:
                eliminarDeteccionesAnidadas(entrada);
                break;

            case ZONAS_ROJAS:
                seleccionBasadaColor(entrada);
                break;
            case DIGITOS_SENAL:
                segmentacionDigitos(entrada);
                break;
        }
        if (mostrarSalida == Salida.SEGMENTACION) {
            return salidasegmentacion;
        }

        // Reconocimiento OCR
        switch (tipoReconocimiento) {
            case SIN_PROCESO:
                salidaocr = entrada;
                break;

            case OCR1:
                salidaocr = ocr1(entrada);
                break;

            case LTROCR:
                salidaocr = leftToRightOCR(entrada);
                break;
        }
        if (mostrarSalida == Salida.RECONOCIMIENTO) {
            return salidaocr;
        }


        salidaCarro = entrada;
        return salidaCarro;
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
        Core.multiply(Gx, Gx, Gx2); //Gx2 = Gx*Gx elemento a elemento
        Core.multiply(Gy, Gy, Gy2); //Gy2 = Gy*Gy elemento a elemento
        Mat ModGrad2 = new Mat();
        Core.add(Gx2, Gy2, ModGrad2);
        Mat ModGrad = new Mat();
        Core.sqrt(ModGrad2, ModGrad);

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
                Size(tam, tam));
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(entrada, gray_dilation, SE); // 3x3 dilation
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, entrada, dilation_residue);

        salidatrlocal = dilation_residue;
    }

    private void binarizacionRojas(Mat entrada) { //Ejemplo para ser rellenado en curso
        Mat red = new Mat();
        Mat green = new Mat();
        Mat blue = new Mat();
        Mat maxGB = new Mat();

        salidabinarizacion = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract(red, maxGB, salidabinarizacion);

        Core.MinMaxLocResult minMax = Core.minMaxLoc(salidabinarizacion);
        int maximum = (int) minMax.maxVal;
        int thresh = maximum / 4;
        Imgproc.threshold(salidabinarizacion, salidabinarizacion, thresh, 255, Imgproc.THRESH_BINARY);
    }

    private void binarizacionOtsu(Mat entrada) {
        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);

        Imgproc.threshold(gris, salidabinarizacion, 0, 255, Imgproc.THRESH_OTSU |
                Imgproc.THRESH_BINARY);
    }

    private void binarizacionAdaptativa(Mat entrada) {
        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);

        //Calculo del gradiente morfológico.
        int contraste = 2;
        int tamano = 7;

        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
                Size(tamano, tamano));
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(gris, gray_dilation, SE); // 3x3 dilation
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, gris, dilation_residue);

        Imgproc.adaptiveThreshold(
                dilation_residue,
                salidabinarizacion,
                255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY,
                tamano,
                -contraste);
    }

    private void seleccionCandidatosCirculos(Mat entrada) {
        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
        Mat binaria = new Mat();

        //Calculo del gradiente morfológico.
        int contraste = 2;
        int tamano = 7;

        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
                Size(tamano, tamano));
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(gris, gray_dilation, SE); // 3x3 dilation
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, gris, dilation_residue);

        Imgproc.adaptiveThreshold(
                dilation_residue,
                binaria,
                255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY,
                tamano,
                -contraste);


        List<MatOfPoint> blobs = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat salida = binaria.clone();//Copia porque finContours modifica entrada
        Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 30;
        float maxratio = (float) 0.75;
        // Seleccionar candidatos a circulos
        for (int c = 0; c < blobs.size(); c++) {
            double[] data = hierarchy.get(0, c);
            int parent = (int) data[3];
            if (parent < 0) //Contorno exterior: rechazar
                continue;
            Rect BB = Imgproc.boundingRect(blobs.get(c));
            // Comprobar tamaño
            if (BB.width < minimumHeight || BB.height < minimumHeight)
                continue;
            // Comprobar anchura similar a altura
            float wf = BB.width;
            float hf = BB.height;
            float ratio = wf / hf;
            if (ratio < maxratio || ratio > 1.0 / maxratio)
                continue;
            // Comprobar no está cerca del borde
            if (BB.x < 2 || BB.y < 2)
                continue;
            if (entrada.width() - (BB.x + BB.width) < 3 || entrada.height() -
                    (BB.y + BB.height) < 3)
                continue;
            // Aqui cumple todos los criterios. Dibujamos
            final Point P1 = new Point(BB.x, BB.y);
            final Point P2 = new Point(BB.x + BB.width, BB.y + BB.height);
            Imgproc.rectangle(salida, P1, P2, new Scalar(255, 0, 0));
        } // for

        salidasegmentacion = salida;

    }

    private void eliminarDeteccionesAnidadas(Mat entrada) {
        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
        Mat binaria = new Mat();

        //Calculo del gradiente morfológico.
        int contraste = 2;
        int tamano = 7;

        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
                Size(tamano, tamano));
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(gris, gray_dilation, SE); // 3x3 dilation
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, gris, dilation_residue);

        Imgproc.adaptiveThreshold(
                dilation_residue,
                binaria,
                255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY,
                tamano,
                -contraste);


        List<MatOfPoint> blobs = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat salida = binaria.clone();//Copia porque finContours modifica entrada
        Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 30;
        float maxratio = (float) 0.75;
        // Seleccionar candidatos a circulos
        for (int c = 0; c < blobs.size(); c++) {
            double[] data = hierarchy.get(0, c);
            int parent = (int) data[3];
            if (parent < 0) //Contorno exterior: rechazar
                continue;
            Rect BB = Imgproc.boundingRect(blobs.get(c));
            // Comprobar tamaño
            if (BB.width < minimumHeight || BB.height < minimumHeight)
                continue;
            // Comprobar anchura similar a altura
            float wf = BB.width;
            float hf = BB.height;
            float ratio = wf / hf;
            if (ratio < maxratio || ratio > 1.0 / maxratio)
                continue;
            // Comprobar no está cerca del borde
            if (BB.x < 2 || BB.y < 2)
                continue;
            if (entrada.width() - (BB.x + BB.width) < 3 || entrada.height() -
                    (BB.y + BB.height) < 3)
                continue;

            // Comprobar que es un círculo
            double minMaxDistanceRatio = minMaxDistanceRatio(blobs.get(c));
            if (0.80 > minMaxDistanceRatio || minMaxDistanceRatio > 1.20)
                continue;

            // Comprobar que es no es exterior
            if (hasInnerCircle(blobs, c))
                continue;

            // Aqui cumple todos los criterios. Dibujamos
            final Point P1 = new Point(BB.x, BB.y);
            final Point P2 = new Point(BB.x + BB.width, BB.y + BB.height);
            Imgproc.rectangle(salida, P1, P2, new Scalar(255, 0, 0));
        } // for

        salidasegmentacion = salida;

    }

    private double minMaxDistanceRatio(MatOfPoint currBlob) {

        Size sizeA = currBlob.size();
        double contourPointsCount = sizeA.width * sizeA.height;
        Point center = centerOf(currBlob);

        double minDistance = contourPointsCount, maxDistance = 0.0;
        for (int i = 0; i < sizeA.height; i++) {
            for (int j = 0; j < sizeA.width; j++) {
                double[] pp = currBlob.get(i, j);
                Point point = new Point(pp[0], pp[1]);
                double distance = euclideanDistance(center, point);
                if (minDistance > distance) minDistance = distance;

                if (maxDistance < distance) maxDistance = distance;
            }
        }

        return minDistance / maxDistance;
    }

    private Point centerOf(MatOfPoint currBlob) {
        Point sum = new Point(0.0, 0.0);
        Size sizeA = currBlob.size();
        for (int i = 0; i < sizeA.height; i++) {
            for (int j = 0; j < sizeA.width; j++) {
                double[] pp = currBlob.get(i, j);
                sum.x += pp[0];
                sum.y += pp[1];
            }
        }
        double contourPointsCount = sizeA.width * sizeA.height;
        sum.x /= contourPointsCount;
        sum.y /= contourPointsCount;

        return sum.clone();
    }

    private double euclideanDistance(Point a, Point b) {
        double distance = 0.0;
        try {
            if (a != null && b != null) {
                double xDiff = a.x - b.x;
                double yDiff = a.y - b.y;
                distance = Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
            }
        } catch (Exception e) {
            System.err.println("Something went wrong in euclideanDistance function in: " + e.getMessage());
        }
        return distance;
    }

    private boolean hasInnerCircle(List<MatOfPoint> allBlobs, int currBlobIndex) {
        MatOfPoint currBlob = allBlobs.get(currBlobIndex);
        Point centerOfCurrent = centerOf(allBlobs.get(currBlobIndex));

        for (int i = 0; i < allBlobs.size(); i++) {
            MatOfPoint blob = allBlobs.get(i);
            // Comparo Puntos Centrales
            if (euclideanDistance(centerOfCurrent, centerOf(blob)) > 5) {
                continue;
            }
            // Busco el que tiene distancia a contorno más peq.
            if (currBlob.size().area() > blob.size().area())
                return true;
        }
        return false;
    }

    private void seleccionBasadaColor(Mat entrada) {
        Mat red = new Mat();
        Mat green = new Mat();
        Mat blue = new Mat();
        Mat maxGB = new Mat();

        Mat binaria = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract(red, maxGB, binaria);

        Core.MinMaxLocResult minMax = Core.minMaxLoc(binaria);
        int maximum = (int) minMax.maxVal;
        int thresh = maximum / 4;
        Imgproc.threshold(binaria, binaria, thresh, 255, Imgproc.THRESH_BINARY);

        List<MatOfPoint> blobs = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat salida = binaria.clone();//Copia porque finContours modifica entrada
        Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 30;
        float maxratio = (float) 0.75;
        // Seleccionar candidatos a circulos
        for (int c = 0; c < blobs.size(); c++) {
            double[] data = hierarchy.get(0, c);
            int parent = (int) data[3];
            if (parent < 0) //Contorno exterior: rechazar
                continue;
            Rect BB = Imgproc.boundingRect(blobs.get(c));
            // Comprobar tamaño
            if (BB.width < minimumHeight || BB.height < minimumHeight)
                continue;
            // Comprobar anchura similar a altura
            float wf = BB.width;
            float hf = BB.height;
            float ratio = wf / hf;
            if (ratio < maxratio || ratio > 1.0 / maxratio)
                continue;
            // Comprobar no está cerca del borde
            if (BB.x < 2 || BB.y < 2)
                continue;
            if (entrada.width() - (BB.x + BB.width) < 3 || entrada.height() -
                    (BB.y + BB.height) < 3)
                continue;

            // Comprobar que es un círculo
            double minMaxDistanceRatio = minMaxDistanceRatio(blobs.get(c));
            if (0.80 > minMaxDistanceRatio || minMaxDistanceRatio > 1.20)
                continue;

            // Comprobar que es no es exterior
            if (hasInnerCircle(blobs, c))
                continue;

            // Aqui cumple todos los criterios. Dibujamos
            final Point P1 = new Point(BB.x, BB.y);
            final Point P2 = new Point(BB.x + BB.width, BB.y + BB.height);
            Imgproc.rectangle(salida, P1, P2, new Scalar(0, 0, 255));
        } // for

        salidasegmentacion = salida;
    }

    private void segmentacionDigitos(Mat entrada) {
        Rect rectCirculo = localizarCirculoRojo(entrada);
        if (rectCirculo == null)
            salidasegmentacion = entrada.clone();
        salidasegmentacion = segmentarInteriorDisco(entrada, rectCirculo);
    }

    private Rect localizarCirculoRojo(Mat entrada) {
        Rect rectCirculo = new Rect();

        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
        Mat binaria = new Mat();

        //Calculo del gradiente morfológico.
        int contraste = 2;
        int tamano = 7;

        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
                Size(tamano, tamano));
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(gris, gray_dilation, SE); // 3x3 dilation
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, gris, dilation_residue);

        Imgproc.adaptiveThreshold(
                dilation_residue,
                binaria,
                255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY,
                tamano,
                -contraste);


        List<MatOfPoint> blobs = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat salida = binaria.clone();//Copia porque finContours modifica entrada
        Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 30;
        float maxratio = (float) 0.75;
        // Seleccionar candidatos a circulos
        for (int c = 0; c < blobs.size(); c++) {
            double[] data = hierarchy.get(0, c);
            int parent = (int) data[3];
            if (parent < 0) //Contorno exterior: rechazar
                continue;
            Rect BB = Imgproc.boundingRect(blobs.get(c));
            // Comprobar tamaño
            if (BB.width < minimumHeight || BB.height < minimumHeight)
                continue;
            // Comprobar anchura similar a altura
            float wf = BB.width;
            float hf = BB.height;
            float ratio = wf / hf;
            if (ratio < maxratio || ratio > 1.0 / maxratio)
                continue;
            // Comprobar no está cerca del borde
            if (BB.x < 2 || BB.y < 2)
                continue;
            if (entrada.width() - (BB.x + BB.width) < 3 || entrada.height() -
                    (BB.y + BB.height) < 3)
                continue;

            // Comprobar que es un círculo
            double minMaxDistanceRatio = minMaxDistanceRatio(blobs.get(c));
            if (0.80 > minMaxDistanceRatio || minMaxDistanceRatio > 1.20)
                continue;

            // Comprobar que es no es exterior
            if (hasInnerCircle(blobs, c))
                continue;

            if (BB.width > rectCirculo.width || BB.height > rectCirculo.height) {
                rectCirculo = BB;
            }
        } // for

        // Release
        gray_dilation.release();
        dilation_residue.release();
        hierarchy.release();
        salida.release();

        return rectCirculo;
    }

    private Mat segmentarInteriorDisco(Mat entrada, Rect rectCirculo) {
        Mat salida = entrada.clone();

        // Extraer componente rojo.
        Mat circRojo = salida.submat(rectCirculo);
        Mat red = new Mat();
        Core.extractChannel(circRojo, red, 0);

        // Binarización Otsu.
        Mat binaria = new Mat();
        Imgproc.threshold(red, binaria, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        // Segmentación de dígitos
        List<MatOfPoint> blobs = new ArrayList<>();
        Mat hierarchy = new Mat();//Copia porque findContours modifica entrada
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 12;

        for (int current = 0; current < blobs.size(); current++) {
            Rect BB = Imgproc.boundingRect(blobs.get(current));

            // Comprobar tamaño mayor a 12 px
            if (BB.height < minimumHeight)
                continue;

            // Comprobar altura mayor que la tercera parte del círculo.
            if (BB.height < rectCirculo.height / 3)
                continue;

            // Comprobar altura mayor que su anchura.
            if (BB.height <= BB.width)
                continue;

            // Comprobar no está cerca del borde
            if (BB.x < 2 || BB.y < 2)
                continue;
            if (circRojo.width() - (BB.x + BB.width) < 3 || circRojo.height() -
                    (BB.y + BB.height) < 3)
                continue;

            // Aqui cumple todos los criterios. Dibujamos
            final Point P1 = new Point(BB.x, BB.y);
            final Point P2 = new Point(BB.x + BB.width, BB.y + BB.height);
            Imgproc.rectangle(circRojo, P1, P2, new Scalar(0, 255, 0));
        }

        //Releases
        binaria.release();
        red.release();
        hierarchy.release();
//        circRojo.release();

        return salida;
    }

    private Mat tabla_caracteristicas;
    private static int NUMERO_CARACTERISTICAS = 9;
    private static int MUESTRAS_POR_CLASE = 2;
    private static int NUMERO_CLASES = 10;
    private Mat binaria1;
    private Mat binaria2;

    private Mat ocr1(Mat entrada) {
        crearTabla();

        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
        Rect rect_digito = new Rect();
        boolean localizado = localizarCaracter(rect_digito);
        if (!localizado)
            return entrada.clone();
        //Recortar rectangulo en imagen original
        Mat recorte_digito = gris.submat(rect_digito);

        //Binarizacion Otsu
        Imgproc.threshold(recorte_digito, binaria2, 0, 255,
                Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        //Leer numero
        int digito = leerRectangulo(binaria2);
        return dibujarResultado(entrada, rect_digito, digito);
    }

    private void crearTabla() {
        double[][] datosEntrenamiento = new double[][]{
                new double[]{0.5757916569709778, 0.8068438172340393,
                        0.6094995737075806, 0.6842694878578186, 0, 0.6750765442848206,
                        0.573646605014801, 0.814811110496521, 0.6094995737075806},
                new double[]{0.5408163070678711, 0.04897959157824516, 0,
                        0.8428571224212646, 0.79795902967453, 0.7795917987823486,
                        0.9938775897026062, 1, 0.995918333530426},
                new double[]{0.7524304986000061, 0.1732638627290726,
                        0.697916567325592, 0.6704860925674438, 0.3805555701255798,
                        0.9767361283302307, 0.6843749284744263, 0.7732638716697693,
                        0.6086806654930115},
                new double[]{0.6724254488945007, 0, 0.6819106936454773,
                        0.6561655402183533, 0.5406503081321716, 0.647357702255249,
                        0.6775066256523132, 0.8231707215309143, 0.732723593711853},
                new double[]{0.02636498026549816, 0.6402361392974854,
                        0.5215936899185181, 0.7385144829750061, 0.5210034847259521,
                        0.6062962412834167, 0.5685194730758667, 0.6251844167709351,
                        0.7910475134849548},
                new double[]{0.8133208155632019, 0.550218939781189,
                        0.6083046793937683, 0.7753458619117737, 0.4955636858940125,
                        0.6764461994171143, 0.4960368871688843, 0.8128473162651062,
                        0.6384715437889099},
                new double[]{0.6108391284942627, 0.985664427280426,
                        0.5884615778923035, 0.7125874161720276, 0.5996503829956055,
                        0.6629370450973511, 0.4828671216964722, 0.7608392238616943,
                        0.6695803999900818},
                new double[]{0.6381308436393738, 0, 0.1727102696895599,
                        0.7140188217163086, 0.5850467085838318, 0.8407476544380188,
                        0.943925142288208, 0.4654205441474915, 0.02728971838951111},
                new double[]{0.6880735158920288, 0.8049609065055847,
                        0.7363235950469971, 0.6299694776535034, 0.672782838344574,
                        0.6411824822425842, 0.6687054634094238, 0.7784574031829834,
                        0.7037037014961243},
                new double[]{0.6497123241424561, 0.7168009877204895,
                        0.4542001485824585, 0.6476410031318665, 0.6150747537612915,
                        0.7033372521400452, 0.5941311717033386, 0.9686998724937439,
                        0.5930955410003662},
                new double[]{0.6764705777168274, 1, 0.7450980544090271,
                        0.7091502547264099, 0.05228758603334427, 0.6993464231491089,
                        0.6339869499206543, 0.9934640526771545, 0.7058823704719543},
                new double[]{0.3452012538909912, 0.3885449171066284, 0,
                        0.7770897746086121, 0.6501547694206238, 0.5789474248886108, 1, 1, 1},
                new double[]{0.6407563090324402, 0.06722689419984818,
                        0.7825630307197571, 0.7132352590560913, 0.6365545988082886,
                        0.9222689270973206, 0.7226890921592712, 0.5850840210914612,
                        0.7058823704719543}, new double[]{0.5980392098426819, 0, 0.6666666865348816,
                0.686274528503418, 0.5751633644104004, 0.6111111640930176,
                0.6111112236976624, 0.7516340017318726, 0.7647058963775635},
                new double[]{0.03549695760011673, 0.717038631439209,
                        0.4705882370471954, 0.7474644780158997, 0.7109533548355103,
                        0.6531440615653992, 0.5862069725990295, 0.6744422316551208,
                        0.780933141708374},
                new double[]{0.6201297640800476, 0.5129870772361755,
                        0.5876624584197998, 0.7207792997360229, 0.5844155550003052,
                        0.6168831586837769, 0.5389610528945923, 0.8214285969734192,
                        0.7435064911842346},
                new double[]{0.6176470518112183, 1, 0.6764706373214722,
                        0.6699347496032715, 0.601307213306427, 0.6405228972434998,
                        0.5098039507865906, 0.7647058963775635, 0.8039215803146362},
                new double[]{0.7272727489471436, 0.0202020201832056,
                        0.2727272808551788, 0.8383838534355164, 0.8181818127632141,
                        0.7272727489471436, 0.8989898562431335, 0.1616161614656448, 0},
                new double[]{0.6928104758262634, 0.8071895837783813,
                        0.8333333134651184, 0.6764705777168274, 0.7026143074035645,
                        0.6209149956703186, 0.6601307392120361, 0.7712417840957642,
                        0.7941176891326904},
                new double[]{0.7320261597633362, 0.8202614784240723,
                        0.5653595328330994, 0.6503268480300903, 0.5882353186607361,
                        0.6732026338577271, 0.6045752167701721, 0.9869281649589539,
                        0.6339869499206543}};
        for (int i = 0; i < 20; i++)
            tabla_caracteristicas.put(i, 0, datosEntrenamiento[i]);
    }

    boolean localizarCaracter(Rect digit_rect) {
        int contraste = 5;
        int tamano = 7;
        Imgproc.adaptiveThreshold(gris, binaria1, 255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV,
                tamano, contraste);
        List<MatOfPoint> contornos = new ArrayList<MatOfPoint>();
        Mat jerarquia = new Mat();
        Imgproc.findContours(binaria1, contornos, jerarquia,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        int altura_minima = 30;
        int anchura_minima = 10;
        int max_area = -1;
        // Seleccionar objeto mas grande
        for (int c = 0; c < contornos.size(); c++) {
            Rect bb = Imgproc.boundingRect(contornos.get(c));
            // Comprobar tamaño
            if (bb.width < anchura_minima || bb.height < altura_minima)
                continue;
            // Descartar proximos al borde
            if (bb.x < 2 || bb.y < 2)
                continue;
            if (binaria1.width() - (bb.x + bb.width) < 3 || binaria1.height() -
                    (bb.y + bb.width) < 3)
                continue;
            // Seleccionar el mayor
            int area = bb.width * bb.height;
            if (area > max_area) {
                max_area = area;
                digit_rect.x = bb.x;
                digit_rect.y = bb.y;
                digit_rect.width = bb.width;
                digit_rect.height = bb.height;
            }
        }
        if (max_area < 0) return false; //No se ha detectado objeto valido
        else return true;
    }

    public int leerRectangulo(Mat rectangulo) {
        Mat vectorCaracteristicas = caracteristicas(rectangulo);
        // Buscamos la fila de la tabla que mas se parece
        double Sumvv = vectorCaracteristicas.dot(vectorCaracteristicas);
        int nmin = 0;
        double Sumvd = tabla_caracteristicas.row(nmin).dot(
                vectorCaracteristicas);
        double Sumdd = tabla_caracteristicas.row(nmin).dot(
                tabla_caracteristicas.row(nmin));
        double D = Sumvd / Math.sqrt(Sumvv * Sumdd);
        double dmin = D;
        for (int n = 1; n < tabla_caracteristicas.rows(); n++) {
            Sumvd = tabla_caracteristicas.row(n).dot(vectorCaracteristicas);
            Sumdd = tabla_caracteristicas.row(n).dot(
                    tabla_caracteristicas.row(n));
            D = Sumvd / Math.sqrt(Sumvv * Sumdd);
            if (D > dmin) {
                dmin = D;
                nmin = n;
            }
        }
        nmin = nmin % 10; // A partir de la fila determinamos el numero
        return nmin;
    }

    private Mat caracteristicas(Mat recorteDigito) {
        //rectangulo: imagen binaria de digito
        //Convertimos a flotante doble precisión
        Mat chardouble = new Mat();
        recorteDigito.convertTo(chardouble, CvType.CV_64FC1);
        //Calculamos vector de caracteristicas
        Mat digito_3x3 = new Mat();
        Imgproc.resize(chardouble, digito_3x3, new Size(3, 3), 0, 0,
                Imgproc.INTER_AREA);
        // convertimos de 3x3 a 1x9 en el orden adecuado
        digito_3x3 = digito_3x3.t();
        return digito_3x3.reshape(1, 1);
    }

    private Mat dibujarResultado(Mat imagen, Rect digit_rect, String digit) {
        Mat salida = imagen.clone();
        Point P1 = new Point(digit_rect.x, digit_rect.y);
        Point P2 = new Point(digit_rect.x + digit_rect.width,
                digit_rect.y + digit_rect.height);
        Imgproc.rectangle(salida, P1, P2, new Scalar(255, 0, 0));
        // Escribir numero
        int fontFace = 6;//FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = 1;
        int thickness = 5;
        Imgproc.putText(salida, digit,
                P1, fontFace, fontScale,
                new Scalar(0, 0, 0), thickness, 8, false);
        Imgproc.putText(salida, digit,
                P1, fontFace, fontScale,
                new Scalar(255, 255, 255), thickness / 2, 8, false);
        return salida;
    }

    private Mat leftToRightOCR(Mat entrada) {
        Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
        Rect rect_circulo = localizarCirculoRojo(entrada);
        if (rect_circulo == null)
            return entrada.clone();
        Mat circulo = entrada.submat(rect_circulo); //Recorte zona de interes
        String cadenaDigitos = analizarInteriorDisco(circulo, rect_circulo);
        if (cadenaDigitos.length() == 0)
            return entrada.clone();
        return dibujarResultado(entrada, rect_circulo, cadenaDigitos);
    }

    private String analizarInteriorDisco(Mat entrada, Rect rect_circulo) {
        StringBuilder cadena = new StringBuilder();
        crearTabla();

        // Extraer componente rojo.
        Mat red = new Mat();
        Core.extractChannel(entrada, red, 0);

        // Binarización Otsu.
        Mat binaria = new Mat();
        Imgproc.threshold(red, binaria, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        // Segmentación de dígitos
        List<MatOfPoint> blobs = new ArrayList<>();
        Mat hierarchy = new Mat();//Copia porque findContours modifica entrada
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 12;

        for (int current = 0; current < blobs.size(); current++) {
            Rect BB = Imgproc.boundingRect(blobs.get(current));

            // Comprobar tamaño mayor a 12 px
            if (BB.height < minimumHeight)
                continue;

            // Comprobar altura mayor que la tercera parte del círculo.
            if (BB.height < rect_circulo.height / 3)
                continue;

            // Comprobar altura mayor que su anchura.
            if (BB.height <= BB.width)
                continue;

            // Comprobar no está cerca del borde
            if (BB.x < 2 || BB.y < 2)
                continue;
            if (entrada.width() - (BB.x + BB.width) < 3 || entrada.height() -
                    (BB.y + BB.height) < 3)
                continue;

            //Recortar rectangulo en imagen original
            Mat recorte_digito = gris.submat(BB);

            //Binarizacion Otsu
            Imgproc.threshold(recorte_digito, binaria2, 0, 255,
                    Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            //Leer numero
            int digito = leerRectangulo(binaria2);
            cadena.append(digito);
        }

        //Releases
        binaria.release();
        red.release();
        hierarchy.release();

        return cadena.toString();
    }

    private Mat dibujarResultado(Mat imagen, Rect digit_rect, int digit) {
        Mat salida = imagen.clone();
        Point P1 = new Point(digit_rect.x, digit_rect.y);
        Point P2 = new Point(digit_rect.x + digit_rect.width,
                digit_rect.y + digit_rect.height);
        Imgproc.rectangle(salida, P1, P2, new Scalar(255, 0, 0));
        // Escribir numero
        int fontFace = 6;//FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = 1;
        int thickness = 5;
        Imgproc.putText(salida, Integer.toString(digit),
                P1, fontFace, fontScale,
                new Scalar(0, 0, 0), thickness, 8, false);
        Imgproc.putText(salida, Integer.toString(digit),
                P1, fontFace, fontScale,
                new Scalar(255, 255, 255), thickness / 2, 8, false);
        return salida;
    }
}


