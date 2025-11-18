using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DetectLine
{
    public static class AngleDetector
    {
        public static List<List<double>> DetectLines(string imagePath, bool isHorizontalLine = true)  // isHorizontalLine - whether to use horizontal line or vertical line for linearity of the L shape object
        {
            List<List<double>> result = new List<List<double>>();  // each List<double> holds a line equation of the form y = a * x + b, for vertical case it is x = b; so the list is either two item length or one.

            Mat srcImage = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            if (srcImage.Empty())
            {
                Console.WriteLine($"The image is empty! No image in path: {imagePath}");
                return new List<List<double>>();
            }

            Mat tempImage = new Mat();
            Cv2.GaussianBlur(srcImage, tempImage, new Size(3, 3), 0);
            //Cv2.Threshold(tempImage, tempImage, 0, 120, ThresholdTypes.Binary | ThresholdTypes.Otsu);
            //Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(9, 9));
            //Cv2.MorphologyEx(tempImage, tempImage, MorphTypes.Open, kernel);
            //saveImage(tempImage, imagePath, "binary");
            Cv2.GaussianBlur(tempImage, tempImage, new Size(5, 5), 0);

            // Edge detect
            Mat edgeImage = new Mat();
            Cv2.Canny(tempImage, edgeImage, 50, 150);
            Cv2.FindContours(edgeImage, out var contours,
                        out _, RetrievalModes.External,
                        ContourApproximationModes.ApproxNone);

            // Tunable parameters (adapt by image size if needed)
            int imageWidth = srcImage.Cols;
            int imageHeight = srcImage.Rows;
            double minSegLen = Math.Max(8.0, Math.Min(imageWidth, imageHeight) * 0.02);   // ignore polygon edges shorter than this
            int minPtsForFit = Math.Max(30, (int)(Math.Min(imageWidth, imageHeight) * 0.03)); // require this many unique points to fit
            double horizDegThresh = 30.0;  // <=30° considered horizontal after mapping
            double vertDegThresh = 30.0;   // <=30° from vertical => mapped angle >= (90-30)=60 ; we use mapped angle > (90-vertThresh)
            double horizRadThresh = horizDegThresh * Math.PI / 180.0;
            double vertRadThresh = vertDegThresh * Math.PI / 180.0;

            var resultImage = new Mat();
            Cv2.CvtColor(srcImage, resultImage, ColorConversionCodes.GRAY2BGR);
            for (int i = 0; i < contours.Length; i++)   // Draw contours in blue to check it
            {
                Cv2.DrawContours(resultImage, contours, i, Scalar.Blue, 1, LineTypes.AntiAlias);
            }
            saveImage(edgeImage, imagePath, "edge");
            saveImage(resultImage, imagePath, "contour");

            // We'll collect fitted lines (vx, vy, x0, y0, count) to draw later
            var fittedLines = new List<(double vx, double vy, double x0, double y0, string type, int ptsCount)>();
            var colorsList = new List<Scalar> { Scalar.Yellow, Scalar.Aqua, Scalar.Red, Scalar.Blue };
            foreach (var contour in contours)
            {
                if (contour.Length < 30)      // skip tiny noise
                    continue;


                // Approximate to reduce number of points
                var approx = Cv2.ApproxPolyDP(contour, 2, true);

                var horizontalPts = new List<Point>();
                var verticalPts = new List<Point>();

                // Examine every consecutive pair of points in the polygon
                for (int i = 0; i < approx.Length; i++)
                {
                    Point pt1 = approx[i];
                    Point pt2 = approx[(i + 1) % approx.Length];
                    double dx = pt2.X - pt1.X;
                    double dy = pt2.Y - pt1.Y;
                    double segLen = Math.Sqrt(dx * dx + dy * dy);
                    if (segLen < minSegLen)
                        continue; // skip short approx edges (likely noise / small corner detail)


                    // compute mapped angle in [0, PI/2]
                    double ang = Math.Atan2(dy, dx);
                    double angMapped = AngleTo0_90Rad(ang); // 0..PI/2

                    // angleMapped small -> horizontal; angleMapped near PI/2 -> vertical
                    bool isHorizontal = angMapped <= horizRadThresh;
                    bool isVertical = angMapped >= (Math.PI / 2.0 - vertRadThresh);

                    // find indices of p1 and p2 in original contour (should exist)
                    int pt1Index = Array.FindIndex(contour, pt => pt.X == pt1.X && pt.Y == pt1.Y);
                    int pt2Index = Array.FindIndex(contour, pt => pt.X == pt2.X && pt.Y == pt2.Y);

                    // collect contour points along the segment between idx1 and idx2 (wrap if needed)
                    List<Point> segPoints = new List<Point>();
                    if (pt1Index >= 0 && pt2Index >= 0)
                    {
                        if (pt2Index >= pt1Index)
                        {
                            for (int k = pt1Index; k <= pt2Index; k++) segPoints.Add(contour[k]);
                        }
                    }
                    else
                    {
                        // fallback: use the approx endpoints if mapping failed
                        segPoints.Add(pt1);
                        segPoints.Add(pt2);
                    }

                    // Add these points to chosen bucket (horizontal/vertical) if classified
                    if (isHorizontal)
                    {
                        horizontalPts.AddRange(segPoints);
                    }
                    if (isVertical)
                    {
                        verticalPts.AddRange(segPoints);
                    }

                    // draw the approx segment (for debugging) - thin cyan
                    Cv2.Line(resultImage, pt1, pt2, new Scalar(200, 200, 0), 1);
                }

                // Remove deduplicate points
                var horizUnique = horizontalPts.Distinct().ToArray();
                var vertUnique = verticalPts.Distinct().ToArray();

                Console.WriteLine($"Contour pts: {contour.Length}, approx edges: {approx.Length}, horizPts={horizUnique.Length}, vertPts={vertUnique.Length}");

                // Fit a line to horizontal points if enough points
                if (horizUnique.Length >= minPtsForFit)
                {
                    var fit = Cv2.FitLine(horizUnique, DistanceTypes.L2, 0, 0.01, 0.01);
                    fittedLines.Add((fit.Vx, fit.Vy, fit.X1, fit.Y1, "H", horizUnique.Length));
                }

                // Fit a line to vertical points if enough points
                if (vertUnique.Length >= minPtsForFit)
                {
                    var fit = Cv2.FitLine(vertUnique, DistanceTypes.L2, 0, 0.01, 0.01);
                    fittedLines.Add((fit.Vx, fit.Vy, fit.X1, fit.Y1, "V", horizUnique.Length));
                }
            }

            // Draw fitted lines across image and print equations
            int idxLine = 1;
            var resultImage2 = new Mat();
            Cv2.CvtColor(srcImage, resultImage2, ColorConversionCodes.GRAY2BGR);
            int colorInd = 0;
            List<double> lineVectors = new List<double>();
            foreach (var L in fittedLines)
            {
                if (isHorizontalLine == true && L.type == "V")   // only horizontal requested
                    continue;
                if (isHorizontalLine == false && L.type == "H")  // only vertical requested
                    continue;

                double vx = L.vx, vy = L.vy, x0 = L.x0, y0 = L.y0;
                lineVectors.Add(vx);
                lineVectors.Add(vy);
                Point p1, p2;

                // avoid division by zero:
                if (Math.Abs(vx) < 1e-8)
                {
                    int x = (int)Math.Round(x0);
                    p1 = new Point(x, 0); p2 = new Point(x, imageHeight - 1);
                }
                else
                {
                    double leftY = y0 + (0 - x0) * (vy / vx);
                    double rightY = y0 + ((imageWidth - 1 - x0) * (vy / vx));
                    p1 = new Point(0, (int)Math.Round(leftY));
                    p2 = new Point(imageWidth - 1, (int)Math.Round(rightY));
                }

                Scalar color = colorsList[colorInd];

                Cv2.Line(resultImage2, p1, p2, color, 1, LineTypes.AntiAlias);

                // print slope/intercept or vertical
                List<double> currentLine = new List<double>();
                if (Math.Abs(vx) < 1e-8)
                {
                    Console.WriteLine($"Line {idxLine++} ({L.type}): vertical x ≈ {x0:F2}, ptsUsed={L.ptsCount}");
                    currentLine.Add(x0);
                }
                else
                {
                    double slope = vy / vx;
                    double intercept = y0 - slope * x0;
                    Console.WriteLine($"Line {idxLine++} ({L.type}): y = {slope:F6} * x + {intercept:F2}, ptsUsed={L.ptsCount}");
                    currentLine.Add(slope);
                    currentLine.Add(intercept);
                }

                result.Add(currentLine);
                colorInd++;
            }

            var angle = CalculateAngle(lineVectors[0], lineVectors[1], lineVectors[2], lineVectors[3]);
            Console.WriteLine($"The angle is: {angle} \n");

            saveImage(resultImage2, imagePath, "detectedLines");
            return result;
        }

        private static double AngleTo0_90Rad(double angleRad)
        {
            // angleRad from atan2(dy,dx) in [-PI, PI]
            if (angleRad < 0) angleRad += Math.PI;   // now in [0, PI)
            if (angleRad >= Math.PI) angleRad -= Math.PI;
            // mirror to [0, PI/2]
            if (angleRad > Math.PI / 2.0) angleRad = Math.PI - angleRad;
            return angleRad; // in [0, PI/2]
        }

        private static void saveImage(Mat imagedst, string imagePath, string tag)
        {
            var filePath = Path.GetDirectoryName(imagePath);
            var fileName = Path.GetFileName(imagePath);
            int increaseInd = 1;
            var temp = fileName.Split('.')[0] + $"_{tag}" + $"_{increaseInd}.png";
            var outputPath = Path.Combine(filePath, "Output");
            if (!Directory.Exists(outputPath))
            {
                Directory.CreateDirectory(outputPath);
            }
            var newFileName = Path.Combine(outputPath, temp);
            while (File.Exists(newFileName))
            {
                temp = fileName.Split('.')[0] + $"_{++increaseInd}.png";
                newFileName = Path.Combine(filePath, temp);
            }
            Cv2.ImWrite(newFileName, imagedst);
        }

        /// <summary>
        /// 计算两条直线的夹角（单位：度，范围 0°~90°）
        /// </summary>
        /// <param name="vx1">直线1方向向量x分量</param>
        /// <param name="vy1">直线1方向向量y分量</param>
        /// <param name="vx2">直线2方向向量x分量</param>
        /// <param name="vy2">直线2方向向量y分量</param>
        /// <returns>夹角（度）</returns>
        public static double CalculateAngle(double vx1, double vy1, double vx2, double vy2)
        {
            // 计算方向向量的点积（取绝对值，确保夹角为最小角）
            double dotProduct = Math.Abs(vx1 * vx2 + vy1 * vy2);

            // 计算两个方向向量的模长
            double len1 = Math.Sqrt(vx1 * vx1 + vy1 * vy1);
            double len2 = Math.Sqrt(vx2 * vx2 + vy2 * vy2);

            // 避免除以0（方向向量无效的情况）
            if (len1 < 1e-6 || len2 < 1e-6)
                return 0.0;

            // 计算cosα
            double cosAlpha = dotProduct / (len1 * len2);

            // 处理浮点数精度问题，确保cosTheta在[-1, 1]范围内
            cosAlpha = Math.Max(-1.0, Math.Min(1.0, cosAlpha));

            // 计算弧度并转换为角度
            double angleRadian = Math.Acos(cosAlpha);
            return angleRadian * 180 / Math.PI;
        }
    }
}
