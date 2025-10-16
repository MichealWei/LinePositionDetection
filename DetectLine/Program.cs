using DetectLine;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
class Program
{
    class LinePolar
    {
        public double Rho;
        public double Theta;
        public int Count; // cluster weight
        public LinePolar(double r, double t, int c = 1) { Rho = r; Theta = t; Count = c; }
    }

    static double AngleDiff(double a, double b)
    {
        double d = Math.Abs(a - b);
        return Math.Min(d, Math.PI - d);
    }

    static Point2d? Intersect(LinePolar a, LinePolar b)
    {
        // solve: [cos a  sin a] [x] = [rhoA]
        //        [cos b  sin b] [y]   [rhoB]
        double cosA = Math.Cos(a.Theta), sinA = Math.Sin(a.Theta);
        double cosB = Math.Cos(b.Theta), sinB = Math.Sin(b.Theta);
        double det = cosA * sinB - sinA * cosB; // = sin(thetaB - thetaA)
        if (Math.Abs(det) < 1e-6) return null; // parallel
        double x = (a.Rho * sinB - sinA * b.Rho) / det;
        double y = (-a.Rho * cosB + cosA * b.Rho) / det;
        return new Point2d(x, y);
    }

    static (Point p1, Point p2)? PointsOnImage(LinePolar L, int width, int height)
    {
        double cosT = Math.Cos(L.Theta), sinT = Math.Sin(L.Theta);
        const double eps = 1e-6;
        var pts = new List<Point2d>();

        // If nearly vertical (cos not ~0) compute x = rho/cos
        if (Math.Abs(cosT) > eps && Math.Abs(sinT) > eps)
        {
            double m = -cosT / sinT;
            double b = L.Rho / sinT; // y = m*x + b

            // intersection with x = 0 and x = width
            double y0 = m * 0 + b;
            if (y0 >= 0 && y0 <= height) pts.Add(new Point2d(0, y0));
            double yw = m * width + b;
            if (yw >= 0 && yw <= height) pts.Add(new Point2d(width, yw));

            // intersection with y = 0 and y = height
            if (Math.Abs(m) > 1e-9)
            {
                double x0 = (0 - b) / m;
                if (x0 >= 0 && x0 <= width) pts.Add(new Point2d(x0, 0));
                double xh = (height - b) / m;
                if (xh >= 0 && xh <= width) pts.Add(new Point2d(xh, height));
            }
        }
        else if (Math.Abs(sinT) <= eps) // nearly vertical line: x = rho / cosT
        {
            double x = L.Rho / cosT;
            if (x >= 0 && x <= width) { pts.Add(new Point2d(x, 0)); pts.Add(new Point2d(x, height)); }
        }
        else if (Math.Abs(cosT) <= eps) // nearly horizontal: y = rho / sinT
        {
            double y = L.Rho / sinT;
            if (y >= 0 && y <= height) { pts.Add(new Point2d(0, y)); pts.Add(new Point2d(width, y)); }
        }

        // Need exactly two distinct points to draw
        var uniq = pts.Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y)))
                      .Distinct()
                      .ToList();
        if (uniq.Count >= 2) return (uniq.First(), uniq.Last());
        return null;
    }

    static void Main()
    {
        //string path = "C:\\ImageAlgorithm\\Images\\tilted.png"; // use your path
        string currentDirectory = Environment.CurrentDirectory;
        string path = Path.Combine(currentDirectory, "Images", "Lshape.png");  // change image name here: "tilted.png", "wafer.png", "Lshape.png", "horizontal.png"
                                                                               //string path = Path.Combine("Images", "tilted.png");

        #region  use folder
        string folder = Path.Combine(currentDirectory, "LinearityImages");
        var ptList = LinesDetector.GetContourPointsFromImages(folder, 100, 5);    // use helper class to package things up-clean code
        var linearityErr = LinesDetector.LinearityError(ptList);
        var linearityError = LinesDetector.CalculateLinearityError(ptList);
        var filePath = Path.Combine(folder, "Output", "LinearityPoints.csv");
        LinesDetector.WritePointsToCsv(ptList, filePath);
        #endregion

        //#region use file
        //var res = LinesDetector.DetectLines(path);   // traditional way to handle single picture
        //foreach (var item in res)
        //{
        //    if (item.Count > 1)
        //        Console.WriteLine($"Line: y = {item[0]:F6} * x + {item[1]:F2}");
        //    else
        //        Console.WriteLine($"Line: x = {item[0]:F6}");
        //}

        //var srcImage = Cv2.ImRead(path, ImreadModes.Grayscale);
        //if (srcImage.Empty())
        //{
        //    Console.WriteLine("Cannot read image at " + path);
        //    return;
        //}


        //// Preprocess: blur, threshold, close small gaps
        //Mat blurredImage = new Mat();
        //Cv2.GaussianBlur(srcImage, blurredImage, new Size(5, 5), 0);
        //Mat binaryImage = new Mat();
        //Cv2.Threshold(blurredImage, binaryImage, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        //Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        //Cv2.MorphologyEx(binaryImage, binaryImage, MorphTypes.Close, kernel);

        //// Edge detect
        //Mat edgeImage = new Mat();
        //Cv2.Canny(binaryImage, edgeImage, 50, 150);
        ////Cv2.Canny(bin, edges, 50, 150, 3, false);
        //saveResultImage(edgeImage, path);

        //// Find contours, use contours points for FitLine instead of all edge points
        //Cv2.FindContours(edgeImage, out var contours,
        //                out _, RetrievalModes.External,
        //                ContourApproximationModes.ApproxNone);


        //int imageWidth = srcImage.Cols;
        //int imageHeight = srcImage.Rows;

        //var resultImage = new Mat();
        //Cv2.CvtColor(srcImage, resultImage, ColorConversionCodes.GRAY2BGR);

        //for (int i = 0; i < contours.Length; i++)   // Draw contours in blue to check it
        //{
        //    Cv2.DrawContours(resultImage, contours, i, Scalar.Blue, 1, LineTypes.AntiAlias);
        //}
        //saveResultImage(resultImage, path);
        //Cv2.ImShow("contours", resultImage);


        //// Tunable parameters (adapt by image size if needed)
        //double minSegLen = Math.Max(8.0, Math.Min(imageWidth, imageHeight) * 0.02);   // ignore polygon edges shorter than this
        //int minPtsForFit = Math.Max(30, (int)(Math.Min(imageWidth, imageHeight) * 0.03)); // require this many unique points to fit
        //double horizDegThresh = 30.0;  // <=30° considered horizontal after mapping
        //double vertDegThresh = 30.0;   // <=30° from vertical => mapped angle >= (90-30)=60 ; we use mapped angle > (90-vertThresh)
        //double horizRadThresh = horizDegThresh * Math.PI / 180.0;
        //double vertRadThresh = vertDegThresh * Math.PI / 180.0;

        //// We'll collect fitted lines (vx, vy, x0, y0, count) to draw later
        //var fittedLines = new List<(double vx, double vy, double x0, double y0, string type, int ptsCount)>();
        //var colorsList = new List<Scalar> { Scalar.Yellow, Scalar.Aqua, Scalar.Red, Scalar.Blue };
        //foreach (var contour in contours)
        //{
        //    if (contour.Length < 30)      // skip tiny noise
        //        continue;


        //    // Approximate to reduce number of points
        //    var approx = Cv2.ApproxPolyDP(contour, 2, true);

        //    var horizontalPts = new List<Point>();
        //    var verticalPts = new List<Point>();

        //    // Examine every consecutive pair of points in the polygon
        //    for (int i = 0; i < approx.Length; i++)
        //    {
        //        Point pt1 = approx[i];
        //        Point pt2 = approx[(i + 1) % approx.Length];
        //        double dx = pt2.X - pt1.X;
        //        double dy = pt2.Y - pt1.Y;
        //        double segLen = Math.Sqrt(dx * dx + dy * dy);
        //        if (segLen < minSegLen)
        //            continue; // skip short approx edges (likely noise / small corner detail)


        //        // compute mapped angle in [0, PI/2]
        //        double ang = Math.Atan2(dy, dx);
        //        double angMapped = AngleTo0_90Rad(ang); // 0..PI/2

        //        // angleMapped small -> horizontal; angleMapped near PI/2 -> vertical
        //        bool isHorizontal = angMapped <= horizRadThresh;
        //        bool isVertical = angMapped >= (Math.PI / 2.0 - vertRadThresh);

        //        // find indices of p1 and p2 in original contour (should exist)
        //        int pt1Index = Array.FindIndex(contour, pt => pt.X == pt1.X && pt.Y == pt1.Y);
        //        int pt2Index = Array.FindIndex(contour, pt => pt.X == pt2.X && pt.Y == pt2.Y);

        //        // collect contour points along the segment between idx1 and idx2 (wrap if needed)
        //        List<Point> segPoints = new List<Point>();
        //        if (pt1Index >= 0 && pt2Index >= 0)
        //        {
        //            if (pt2Index >= pt1Index)
        //            {
        //                for (int k = pt1Index; k <= pt2Index; k++) segPoints.Add(contour[k]);
        //            }
        //        }
        //        else
        //        {
        //            // fallback: use the approx endpoints if mapping failed
        //            segPoints.Add(pt1);
        //            segPoints.Add(pt2);
        //        }

        //        // Add these points to chosen bucket (horizontal/vertical) if classified
        //        if (isHorizontal)
        //        {
        //            horizontalPts.AddRange(segPoints);
        //        }
        //        if (isVertical)
        //        {
        //            verticalPts.AddRange(segPoints);
        //        }

        //        // draw the approx segment (for debugging) - thin cyan
        //        Cv2.Line(resultImage, pt1, pt2, new Scalar(200, 200, 0), 1);
        //    }

        //    // Remove deduplicate points
        //    var horizUnique = horizontalPts.Distinct().ToArray();
        //    var vertUnique = verticalPts.Distinct().ToArray();

        //    Console.WriteLine($"Contour pts: {contour.Length}, approx edges: {approx.Length}, horizPts={horizUnique.Length}, vertPts={vertUnique.Length}");

        //    // Fit a line to horizontal points if enough points
        //    if (horizUnique.Length >= minPtsForFit)
        //    {
        //        var fit = Cv2.FitLine(horizUnique, DistanceTypes.L2, 0, 0.01, 0.01);
        //        fittedLines.Add((fit.Vx, fit.Vy, fit.X1, fit.Y1, "H", horizUnique.Length));
        //    }

        //    // Fit a line to vertical points if enough points
        //    if (vertUnique.Length >= minPtsForFit)
        //    {
        //        var fit = Cv2.FitLine(vertUnique, DistanceTypes.L2, 0, 0.01, 0.01);
        //        fittedLines.Add((fit.Vx, fit.Vy, fit.X1, fit.Y1, "V", horizUnique.Length));
        //    }
        //}

        //// Draw fitted lines across image and print equations
        //int idxLine = 1;
        //var resultImage2 = new Mat();
        //Cv2.CvtColor(srcImage, resultImage2, ColorConversionCodes.GRAY2BGR);
        //int colorInd = 0;
        //foreach (var L in fittedLines)
        //{
        //    double vx = L.vx, vy = L.vy, x0 = L.x0, y0 = L.y0;
        //    Point p1, p2;

        //    // avoid division by zero:
        //    if (Math.Abs(vx) < 1e-8)
        //    {
        //        int x = (int)Math.Round(x0);
        //        p1 = new Point(x, 0); p2 = new Point(x, imageHeight - 1);
        //    }
        //    else
        //    {
        //        double leftY = y0 + (0 - x0) * (vy / vx);
        //        double rightY = y0 + ((imageWidth - 1 - x0) * (vy / vx));
        //        p1 = new Point(0, (int)Math.Round(leftY));
        //        p2 = new Point(imageWidth - 1, (int)Math.Round(rightY));
        //    }

        //    Scalar color = colorsList[colorInd];

        //    Cv2.Line(resultImage2, p1, p2, color, 1, LineTypes.AntiAlias);

        //    // print slope/intercept or vertical
        //    if (Math.Abs(vx) < 1e-8)
        //    {
        //        Console.WriteLine($"Line {idxLine++} ({L.type}): vertical x ≈ {x0:F2}, ptsUsed={L.ptsCount}");
        //    }
        //    else
        //    {
        //        double slope = vy / vx;
        //        double intercept = y0 - slope * x0;
        //        Console.WriteLine($"Line {idxLine++} ({L.type}): y = {slope:F6} * x + {intercept:F2}, ptsUsed={L.ptsCount}");
        //    }

        //    colorInd++;
        //}

        //saveResultImage(resultImage2, path);
        //Cv2.ImShow("fit lines with contours", resultImage2);
        //#endregion

        #region fit line with all non-zero points

        //// edge non -zero points for FitLine
        //Mat nonZero = new Mat();
        //Cv2.FindNonZero(edgeImage, nonZero);
        //Point[] edgePoints = Enumerable.Range(0, nonZero.Rows)
        //                               .Select(i => nonZero.At<Point>(i))
        //                               .ToArray();

        //if (!nonZero.Empty())
        //{
        //    // Convert Mat of points to Point[] for FitLine
        //    // Mat.ToArray<T>() returns an array of the specified value type (Point)
        //    Point[] pts = Enumerable.Range(0, nonZero.Rows)
        //                               .Select(i => nonZero.At<Point>(i))
        //                               .ToArray();
        //    var outImg2 = new Mat();
        //    Cv2.CvtColor(srcImage, outImg2, ColorConversionCodes.GRAY2BGR);

        //    if (pts.Length >= 2)
        //    {
        //        // FitLine: returns Vec4f [vx, vy, x0, y0]
        //        var lineVec = Cv2.FitLine(pts, DistanceTypes.L2, 0, 0.01, 0.01);


        //        float vx = (float)lineVec.Vx, vy = (float)lineVec.Vy;
        //        float x0 = (float)lineVec.X1, y0 = (float)lineVec.Y1;

        //        // If direction is nearly vertical (vx ~ 0), compute intersections with top/bottom,
        //        // otherwise compute y at left and right borders.
        //        Point pLeft, pRight;
        //        if (Math.Abs(vx) < 1e-6)
        //        {
        //            // vertical-ish: x ~ x0; draw from top to bottom
        //            int x = (int)Math.Round(x0);
        //            pLeft = new Point(x, 0);
        //            pRight = new Point(x, srcImage.Height - 1);
        //        }
        //        else
        //        {
        //            double leftY = y0 + (0 - x0) * (vy / vx);
        //            double rightY = y0 + (srcImage.Width - 1 - x0) * (vy / vx);
        //            pLeft = new Point(0, (int)Math.Round(leftY));
        //            pRight = new Point(srcImage.Width - 1, (int)Math.Round(rightY));
        //        }

        //        // Draw fitted least-squares line in GREEN (this usually aligns better with stripe center)
        //        Cv2.Line(outImg2, pLeft, pRight, Scalar.Green, 1);
        //        saveResultImage(outImg2, path);
        //        Cv2.ImShow("edges points fitline", outImg2);

        //        // Print fitted line in slope-intercept form if possible
        //        if (Math.Abs(vx) < 1e-6)
        //            Console.WriteLine($"FitLine (vertical): x = {x0:F2}");
        //        else
        //        {
        //            double m = vy / vx;             // slope in param (note: FitLine uses direction (vx,vy))
        //            double b = y0 - m * x0;         // intercept
        //            Console.WriteLine($"FitLine: y = {m:F6} * x + {b:F3}");
        //        }
        //    }
        //}

        #endregion

        #region  Hough line detection
        // Standard Hough (rho,theta)
        //var raw = Cv2.HoughLines(edgeImage, 1, Math.PI / 360.0, 120); // adjust threshold as needed

        //// 3. Prepare color image for drawing
        //var resultImg = new Mat();
        //Cv2.CvtColor(srcImage, resultImg, ColorConversionCodes.GRAY2BGR);

        //// ---- NEW: draw *original* Hough lines in BLUE ----
        //foreach (var v in raw)
        //{
        //    double rho = v.Rho;
        //    double theta = v.Theta;
        //    double a = Math.Cos(theta);    // Horizontal component of the unit normal vector pointing perpendicular to the line
        //    double b = Math.Sin(theta);    // Vertical component of the unit normal vector pointing perpendicular to the line
        //    double x0 = a * rho;           // To draw the line we need two points on it(pt1, pt2). A convenient “base point” is   (x0​,y0​)=(ρa,ρb),
        //    double y0 = b * rho;
        //    Point pt1 = new Point((int)Math.Round(x0 + 1000 * (-b)),         // (-b, a) is the direction along the line.
        //                          (int)Math.Round(y0 + 1000 * (a)));         // Multiplying by ±1000 moves far in both directions to get two points you can pass to Cv2.Line.
        //    Point pt2 = new Point((int)Math.Round(x0 - 1000 * (-b)),
        //                          (int)Math.Round(y0 - 1000 * (a)));
        //    var borderPts = GetBorderPoints(rho, theta, srcImage.Width, srcImage.Height);
        //    Cv2.Line(resultImg, borderPts[0], borderPts[1], new Scalar(255, 0, 0), 1);
        //    //Cv2.Line(resultImg, pt1, pt2, new Scalar(255, 0, 0), 1, LineTypes.AntiAlias);
        //}
        //saveResultImage(resultImg, path);


        //// Cluster/merge similar (rho,theta)
        //double angleThresh = Math.PI / 180.0 * 8.0; // 8 degrees tolerance
        //double rhoThresh = 20.0; // pixels tolerance
        //var clusters = new List<LinePolar>();
        //foreach (var v in raw)
        //{
        //    double rho = v.Rho;
        //    double theta = v.Theta;
        //    bool added = false;
        //    foreach (var c in clusters)
        //    {
        //        if (AngleDiff(theta, c.Theta) < angleThresh && Math.Abs(rho - c.Rho) < rhoThresh)
        //        {
        //            // incremental average
        //            c.Rho = (c.Rho * c.Count + rho) / (c.Count + 1);
        //            // average angle correctly with sin/cos could be done, but simple average is ok here:
        //            c.Theta = (c.Theta * c.Count + theta) / (c.Count + 1);
        //            c.Count++;
        //            added = true;
        //            break;
        //        }
        //    }
        //    if (!added)
        //        clusters.Add(new LinePolar(rho, theta));
        //}

        //// Separate vertical-like and horizontal-like groups
        //double orientThresh = Math.PI / 180.0 * 20.0; // 20 degrees to classify horiz/vert
        //var horizontals = clusters.Where(c => AngleDiff(c.Theta, Math.PI / 2) < orientThresh).ToList();
        //var verticals = clusters.Where(c => AngleDiff(c.Theta, 0) < orientThresh || AngleDiff(c.Theta, Math.PI) < orientThresh).ToList();

        //// If we found more than two in a group, pick the two farthest apart by rho (likely inner/outer edges)
        //LinePolar[] pick2(List<LinePolar> list)
        //{
        //    if (list == null || list.Count == 0) return new LinePolar[0];
        //    if (list.Count == 1) return new[] { list[0] };
        //    var ordered = list.OrderBy(x => x.Rho).ToArray();
        //    return new[] { ordered.First(), ordered.Last() };
        //}

        //var picked = new List<LinePolar>();
        //picked.AddRange(pick2(verticals));
        //picked.AddRange(pick2(horizontals));

        //// fallback: if we haven't got 4, pick the biggest clusters overall (up to 4)
        //if (picked.Count < 4)
        //{
        //    var others = clusters.OrderByDescending(c => c.Count).ToList();
        //    foreach (var o in others)
        //    {
        //        if (!picked.Contains(o)) picked.Add(o);
        //        if (picked.Count == 4) break;
        //    }
        //}

        //// Keep only up to 4 lines
        //picked = picked.Take(4).ToList();

        //// Compute intersection corners (we expect 2 vertical × 2 horizontal => 4 corners)
        //var verts = picked.Where(p => AngleDiff(p.Theta, 0) < orientThresh || AngleDiff(p.Theta, Math.PI) < orientThresh).ToList();
        //var hors = picked.Where(p => AngleDiff(p.Theta, Math.PI / 2) < orientThresh).ToList();
        //var corners = new List<Point2d>();
        //foreach (var v in verts)
        //    foreach (var h in hors)
        //    {
        //        var ip = Intersect(v, h);
        //        if (ip.HasValue) corners.Add(ip.Value);
        //    }

        //Console.WriteLine("Detected lines (up to 4):");
        //int idx = 0;
        //foreach (var L in picked)
        //{
        //    idx++;
        //    // line in y = m x + b (unless near vertical)
        //    double cosT = Math.Cos(L.Theta), sinT = Math.Sin(L.Theta);
        //    if (Math.Abs(sinT) < 1e-6)
        //    {
        //        double x = L.Rho / cosT;
        //        Console.WriteLine($"Line {idx}: vertical x = {x:F2}  (rho={L.Rho:F2}, theta={L.Theta:F3})");
        //    }
        //    else
        //    {
        //        double m = -cosT / sinT;
        //        double b = L.Rho / sinT;
        //        Console.WriteLine($"Line {idx}: y = {m:F6} * x + {b:F3}  (rho={L.Rho:F2}, theta={L.Theta:F3})");
        //    }
        //}

        //if (corners.Count >= 4)
        //{
        //    Console.WriteLine("\nCorner points (intersections):");
        //    foreach (var c in corners) Console.WriteLine($"({c.X:F2}, {c.Y:F2})");
        //}
        //else
        //{
        //    Console.WriteLine("\nCould not get four corners by intersection. Found corners: " + corners.Count);
        //}

        //// Visualize: draw extended lines and corner points
        //Mat outImg = new Mat();
        //Cv2.CvtColor(srcImage, outImg, ColorConversionCodes.GRAY2BGR);

        //foreach (var L in picked)
        //{
        //    var seg = PointsOnImage(L, srcImage.Width - 1, srcImage.Height - 1);
        //    if (seg.HasValue)
        //    {
        //        Cv2.Line(outImg, seg.Value.p1, seg.Value.p2, new Scalar(0, 0, 255), 1);
        //    }
        //    else
        //    {
        //        // fallback: draw a long line via two far points using angle
        //        double cosT = Math.Cos(L.Theta), sinT = Math.Sin(L.Theta);
        //        // find one point on the line: (x0,y0) = (rho*cos, rho*sin)
        //        double x0 = L.Rho * cosT, y0 = L.Rho * sinT;
        //        Point p1 = new Point((int)Math.Round(x0 + 1000 * (-sinT)), (int)Math.Round(y0 + 1000 * cosT));
        //        Point p2 = new Point((int)Math.Round(x0 - 1000 * (-sinT)), (int)Math.Round(y0 - 1000 * cosT));
        //        Cv2.Line(outImg, p1, p2, new Scalar(0, 0, 255), 1);
        //    }
        //}

        //foreach (var c in corners)
        //{
        //    Cv2.Circle(outImg, new Point((int)Math.Round(c.X), (int)Math.Round(c.Y)), 6, new Scalar(0, 255, 0), -1);
        //}
        //saveResultImage(outImg, path);

        //Cv2.ImShow("Detected full lines", outImg);
        #endregion

        Cv2.WaitKey(0);
        Cv2.DestroyAllWindows();
    }

    static void saveResultImage(Mat imagedst, string imagePath)
    {
        var filePath = Path.GetDirectoryName(imagePath);
        var fileName = Path.GetFileName(imagePath);
        int increaseInd = 1;
        var temp = fileName.Split('.')[0] + $"_{increaseInd}.png";
        var newFileName = Path.Combine(filePath, temp);
        while (File.Exists(newFileName))
        {
            temp = fileName.Split('.')[0] + $"_{++increaseInd}.png";
            newFileName = Path.Combine(filePath, temp);
        }
        Cv2.ImWrite(newFileName, imagedst);
    }

    static Point[] GetBorderPoints(double rho, double theta, int width, int height)
    {
        double cosT = Math.Cos(theta);
        double sinT = Math.Sin(theta);
        var pts = new List<Point>();

        // Intersect with left (x=0) and right (x=width-1)
        if (Math.Abs(sinT) > 1e-6)
        {
            int yLeft = (int)Math.Round((rho - 0 * cosT) / sinT);
            int yRight = (int)Math.Round((rho - (width - 1) * cosT) / sinT);
            if (0 <= yLeft && yLeft < height) pts.Add(new Point(0, yLeft));
            if (0 <= yRight && yRight < height) pts.Add(new Point(width - 1, yRight));
        }

        // Intersect with top (y=0) and bottom (y=height-1)
        if (Math.Abs(cosT) > 1e-6)
        {
            int xTop = (int)Math.Round((rho - 0 * sinT) / cosT);
            int xBottom = (int)Math.Round((rho - (height - 1) * sinT) / cosT);
            if (0 <= xTop && xTop < width) pts.Add(new Point(xTop, 0));
            if (0 <= xBottom && xBottom < width) pts.Add(new Point(xBottom, height - 1));
        }

        // Pick two distinct points
        return pts.Count >= 2 ? new[] { pts[0], pts[1] }
                              : new[] { new Point(), new Point() };
    }

    private static void DrawFullLine(Mat img, Line2D fit, int width, int height, Scalar color)
    {
        float vx = (float)fit.Vx, vy = (float)fit.Vy, x0 = (float)fit.X1, y0 = (float)fit.Y1;
        Point p1, p2;
        if (Math.Abs(vx) > Math.Abs(vy)) // horizontal-ish
        {
            int yL = (int)Math.Round(y0 - (x0 * vy / vx));
            int yR = (int)Math.Round(y0 + ((width - x0) * vy / vx));
            p1 = new Point(0, yL);
            p2 = new Point(width - 1, yR);
        }
        else                             // vertical-ish
        {
            int xT = (int)Math.Round(x0 - (y0 * vx / vy));
            int xB = (int)Math.Round(x0 + ((height - y0) * vx / vy));
            p1 = new Point(xT, 0);
            p2 = new Point(xB, height - 1);
        }
        Cv2.Line(img, p1, p2, color, 2, LineTypes.AntiAlias);
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

    private static double LinearityError(Point[] contour, Point a, Point b)
    {
        double A = a.Y - b.Y;
        double B = b.X - a.X;
        double C = a.X * b.Y - b.X * a.Y;
        double denom = Math.Sqrt(A * A + B * B);

        double sum = 0, maxErr = 0;
        foreach (var p in contour)
        {
            double dist = Math.Abs(A * p.X + B * p.Y + C) / denom;
            sum += dist;
            if (dist > maxErr) maxErr = dist;
        }

        double meanErr = sum / contour.Length;
        Console.WriteLine($"Mean error = {meanErr:F2}, Max error = {maxErr:F2}");
        return meanErr;
    }

}
