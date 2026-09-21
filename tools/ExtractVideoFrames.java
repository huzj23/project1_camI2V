import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;

/** Extracts representative frames using the FFmpeg libraries bundled with Flashback. */
public final class ExtractVideoFrames {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            throw new IllegalArgumentException("usage: VIDEO OUTPUT_DIR SECOND...");
        }
        File outputDir = new File(args[1]);
        if (!outputDir.isDirectory() && !outputDir.mkdirs()) {
            throw new IllegalStateException("cannot create " + outputDir);
        }
        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(args[0]);
             Java2DFrameConverter converter = new Java2DFrameConverter()) {
            grabber.start();
            for (int i = 2; i < args.length; i++) {
                double seconds = Double.parseDouble(args[i]);
                grabber.setTimestamp(Math.round(seconds * 1_000_000.0));
                Frame frame = grabber.grabImage();
                if (frame == null) {
                    throw new IllegalStateException("no frame at " + seconds + " seconds");
                }
                BufferedImage image = converter.convert(frame);
                File target = new File(outputDir, String.format("frame_%06.3fs.jpg", seconds));
                ImageIO.write(image, "jpg", target);
                System.out.println(target.getAbsolutePath());
            }
            grabber.stop();
        }
    }
}
