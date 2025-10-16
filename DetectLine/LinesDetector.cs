using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Controls;

namespace DetectLine
{
    public static class LinesDetector
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
            Cv2.GaussianBlur(srcImage, tempImage, new Size(5, 5), 0);
            Cv2.Threshold(tempImage, tempImage, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
            Cv2.MorphologyEx(tempImage, tempImage, MorphTypes.Close, kernel);

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
            foreach (var L in fittedLines)
            {
                if(isHorizontalLine == true && L.type == "V")   // only horizontal requested
                    continue;
                if(isHorizontalLine == false && L.type == "H")  // only vertical requested
                    continue;

                double vx = L.vx, vy = L.vy, x0 = L.x0, y0 = L.y0;
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

            saveImage(resultImage2, imagePath, "detectedLines");
            return result;
        }

        /// <summary>
        /// 计算直线度误差（两端点法）
        /// </summary>
        /// <param name="points">检测点集合（至少包含2个点，否则返回0）</param>
        /// <returns>直线度误差（所有点到两端点连线的最大垂直距离）</returns>
        public static double CalculateLinearityError(List<Point> points)
        {
            // 输入验证：至少需要2个点才能构成直线
            if (points == null || points.Count < 2)
            {
                throw new ArgumentException("检测点数量不足，至少需要2个点");
            }

            // 去重处理（避免重复点导致计算异常）
            var distinctPoints = points.Distinct().ToList();
            if (distinctPoints.Count < 2)
            {
                return 0; // 所有点重合，直线度误差为0
            }

            // 取首尾两点作为基准端点（若点集无序，建议先按X轴排序）
            var p1 = distinctPoints.First();
            var p2 = distinctPoints.Last();

            // 计算两端点连线的直线方程参数（Ax + By + C = 0）
            double A = p2.Y - p1.Y;       // A = y2 - y1
            double B = p1.X - p2.X;       // B = x1 - x2
            double C = p2.X * p1.Y - p1.X * p2.Y; // C = x2y1 - x1y2

            // 计算直线长度（用于垂直距离公式的分母，避免重复计算）
            double lineLength = Math.Sqrt(A * A + B * B);
            if (lineLength < 1e-9) // 避免除以0（两点几乎重合）
            {
                return 0;
            }
            //double maxDistance = 0;

            double maxPositive = 0; // 最大正向偏差（直线一侧）
            double maxNegative = 0; // 最大负向偏差（直线另一侧，记录绝对值）
            Point positiveFarPoint = new Point(); // 最大正向偏差点
            Point negativeFarPoint = new Point(); // 最大负向偏差点

            // 遍历所有点，计算到基准直线的垂直距离
            foreach (var point in distinctPoints)
            {
                double x = point.X;
                double y = point.Y;

                //// 点到直线的垂直距离公式：|Ax + By + C| / √(A² + B²)
                //double distance = Math.Abs(A * x + B * y + C) / lineLength;

                //// 记录最大距离（即直线度误差）
                //if (distance > maxDistance)
                //{
                //    maxDistance = distance;
                //}


                // 计算带符号的距离（分子部分，保留正负）
                double signedDistanceNumerator = A * x + B * y + C;
                // 带符号的距离（未除以模长时，符号已能反映方向）
                double signedDistance = signedDistanceNumerator / lineLength;

                // 区分正负偏差
                if (signedDistance > 0)
                {
                    if (signedDistance > maxPositive)
                    {
                        maxPositive = signedDistance;
                        positiveFarPoint = point;
                    }
                }
                else
                {
                    double absNegative = Math.Abs(signedDistance);
                    if (absNegative > maxNegative)
                    {
                        maxNegative = absNegative;
                        negativeFarPoint = point;
                    }
                }
            }

            Console.WriteLine($"最大正向偏差点: [{positiveFarPoint.X},{positiveFarPoint.Y}]; 最大负向偏差点: [{negativeFarPoint.X},{negativeFarPoint.Y}]");
            // 直线度误差 = 两侧最大距离之和
            return maxPositive + maxNegative;
            //return maxDistance;
        }

        public static double LinearityError(List<Point> contour)
        {
            int lastPt = contour.Count - 1;
            double A = contour[0].Y - contour[lastPt].Y;
            double B = contour[lastPt].X - contour[0].X;
            double C = contour[0].X * contour[lastPt].Y - contour[lastPt].X * contour[0].Y;
            double denom = Math.Sqrt(A * A + B * B);

            double sum = 0, maxErr = 0;
            foreach (var p in contour)
            {
                double dist = Math.Abs(A * p.X + B * p.Y + C) / denom;
                sum += dist;
                if (dist > maxErr) maxErr = dist;
            }

            double meanErr = sum / contour.Count;
            Console.WriteLine($"Mean error = {meanErr:F2}, Max error = {maxErr:F2}");
            return meanErr;
        }

        public static List<Point> GetContourPointsFromImages(string folder, int offsetInImage, int imageDistance)
        {
            var allFiles = Directory.GetFiles(folder);
            List < Point > result = new List< Point >();
            if (allFiles.Length  == 0)
            {
                Console.WriteLine($"No image files in {folder}");
                return result;
            }

            var outputPath = Path.Combine(folder, "Output");  // clear result files from last time
            if (Directory.Exists(outputPath))
            {
                Directory.Delete(outputPath, recursive: true);
            }
            

            // 按文件名中 "location-数字" 的数字部分排序
            var sortedFiles = allFiles.OrderBy(file =>
            {
                string fileName = Path.GetFileNameWithoutExtension(file); // 去掉扩展名（如 .txt）
                                                                          // 按 "-" 拆分文件名（例如 "abc-location-10" 拆分为 ["abc", "location", "10"]）
                string[] parts = fileName.Split(new[] { '-' }, StringSplitOptions.RemoveEmptyEntries);
                // 提取最后一段（数字部分），转换为整数用于排序
                if (parts.Length >= 1 && int.TryParse(parts.Last(), out int num))
                {
                    return num; // 按数字大小排序
                }
                // 若无法提取数字，默认排在最后（可根据需求调整）
                return int.MaxValue;
            }).ToArray();

            int imageIndex = 0;
            foreach (var file in sortedFiles)
            {
                var lineResult = LinesDetector.DetectLines(file);
                var firstLineOffset = lineResult[0][0] * offsetInImage + lineResult[0][1];
                var secondLineOffset = lineResult[1][0] * offsetInImage + lineResult[1][1];
                var averageOffset = (firstLineOffset + secondLineOffset) / 2;
                Point point = new Point();
                point.X = imageIndex * imageDistance;
                point.Y = (int)Math.Round(averageOffset);
                result.Add(point);
                imageIndex++;
            }

             return result;
        }


        /// <summary>
        /// 将点集写入CSV文件（格式：X,Y）
        /// </summary>
        /// <param name="points">待写入的点集（每个点包含X、Y坐标）</param>
        /// <param name="filePath">CSV文件保存路径（如：@"C:\data\points.csv"）</param>
        /// <param name="includeHeader">是否包含表头（默认包含，表头为"X,Y"）</param>
        /// <returns>是否写入成功</returns>
        public static bool WritePointsToCsv(List<Point> points, string filePath, bool includeHeader = false)
        {
            // 输入验证
            if (points == null || points.Count == 0)
            {
                Console.WriteLine("错误：点集为空，无需写入");
                return false;
            }

            if (string.IsNullOrEmpty(filePath))
            {
                Console.WriteLine("错误：文件路径不能为空");
                return false;
            }

            try
            {
                // 创建文件目录（若不存在）
                string directory = Path.GetDirectoryName(filePath);
                if (!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                // 写入CSV内容
                using (StreamWriter writer = new StreamWriter(filePath, false, Encoding.UTF8))
                {
                    // 写入表头（可选）
                    if (includeHeader)
                    {
                        writer.WriteLine("X,Y");
                    }

                    // 写入每个点的坐标（格式：X值,Y值，保留6位小数）
                    foreach (var point in points)
                    {
                        // 格式化坐标为字符串（避免科学计数法，保留6位小数）
                        string line = $"[{point.X:F6},{point.Y:F6}]";
                        writer.WriteLine(line);
                    }
                }

                Console.WriteLine($"成功写入CSV文件：{filePath}，共 {points.Count} 个点");
                return true;
            }
            catch (UnauthorizedAccessException ex)
            {
                Console.WriteLine($"权限错误：无法写入文件 {filePath}，{ex.Message}");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"IO错误：文件操作失败，{ex.Message}（可能文件被占用）");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"写入失败：{ex.Message}");
            }

            return false;
        }

        // Map any angle to [0, PI) then to [0, PI/2] so 0 and ~PI are both horizontal
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

 
}
}
