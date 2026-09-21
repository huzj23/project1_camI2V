package io.v2dit.camera;

import com.moulberry.flashback.exporting.ExportSettings;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.world.phys.Vec3;
import org.joml.Quaternionf;
import org.joml.Vector3fc;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

public final class CameraCsvRecorder {
    private enum Mode { OFF, MANUAL, FLASHBACK_EXPORT }

    private static final Object LOCK = new Object();
    private static final DateTimeFormatter SESSION_TIME = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");
    private static final String HEADER = String.join(",",
            "schema_version", "session_id", "mode", "frame_index", "wall_time_ns",
            "x", "y", "z", "yaw_deg", "pitch_deg",
            "quat_x", "quat_y", "quat_z", "quat_w",
            "right_x", "right_y", "right_z",
            "up_x", "up_y", "up_z",
            "back_x", "back_y", "back_z",
            "fov_deg", "width", "height", "dimension", "source"
    );

    private static Mode mode = Mode.OFF;
    private static BufferedWriter writer;
    private static String sessionId;
    private static long frameIndex;
    private static int exportWidth;
    private static int exportHeight;

    private CameraCsvRecorder() {
    }

    public static boolean isManualRecording() {
        synchronized (LOCK) {
            return mode == Mode.MANUAL;
        }
    }

    public static boolean isExportRecording() {
        synchronized (LOCK) {
            return mode == Mode.FLASHBACK_EXPORT;
        }
    }

    public static boolean startManual() {
        return start(Mode.MANUAL, null);
    }

    public static boolean startFlashbackExport(ExportSettings settings) {
        return start(Mode.FLASHBACK_EXPORT, settings);
    }

    private static boolean start(Mode requestedMode, ExportSettings settings) {
        synchronized (LOCK) {
            stopLocked();
            Minecraft client = Minecraft.getInstance();
            sessionId = SESSION_TIME.format(LocalDateTime.now()) + "-" + requestedMode.name().toLowerCase(Locale.ROOT);
            Path trajectoryDir = client.gameDirectory.toPath().resolve("v2dit_capture").resolve("trajectory");
            Path csv = trajectoryDir.resolve(sessionId + ".csv");
            Path metadata = trajectoryDir.resolve(sessionId + ".metadata.json");
            try {
                Files.createDirectories(trajectoryDir);
                writer = Files.newBufferedWriter(csv, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
                writer.write(HEADER);
                writer.newLine();
                writer.flush();
                writeMetadata(metadata, requestedMode, settings, csv.getFileName().toString());
                frameIndex = 0;
                exportWidth = settings == null ? 0 : settings.resolutionX();
                exportHeight = settings == null ? 0 : settings.resolutionY();
                mode = requestedMode;
                return true;
            } catch (Exception error) {
                System.err.println("[V2DiT Camera Logger] Could not start recording: " + error.getMessage());
                stopLocked();
                try {
                    Files.deleteIfExists(csv);
                    Files.deleteIfExists(metadata);
                } catch (IOException cleanupError) {
                    System.err.println("[V2DiT Camera Logger] Could not remove incomplete session files: "
                            + cleanupError.getMessage());
                }
                return false;
            }
        }
    }

    private static void writeMetadata(Path metadata, Mode requestedMode, ExportSettings settings, String csvName)
            throws IOException {
        String exportDetails;
        if (settings == null) {
            exportDetails = "null";
        } else {
            exportDetails = String.format(Locale.ROOT,
                    "{\n    \"name\": \"%s\",\n    \"output\": \"%s\",\n    \"fps\": %.8f,\n" +
                            "    \"width\": %d,\n    \"height\": %d,\n    \"start_tick\": %d,\n" +
                            "    \"end_tick\": %d,\n    \"projection\": \"%s\",\n    \"depth_map\": %s,\n" +
                            "    \"ssaa\": %s\n  }",
                    JsonText.escape(settings.name()), JsonText.escape(String.valueOf(settings.output())), settings.framerate(),
                    settings.resolutionX(), settings.resolutionY(), settings.startTick(), settings.endTick(),
                    JsonText.escape(String.valueOf(settings.projection())), settings.depthMap(), settings.ssaa());
        }

        String json = String.format(Locale.ROOT,
                "{\n  \"schema_version\": 1,\n  \"session_id\": \"%s\",\n  \"mode\": \"%s\",\n" +
                        "  \"csv\": \"%s\",\n  \"coordinate_system\": {\n" +
                        "    \"world\": \"Minecraft: +X east, +Y up, +Z south\",\n" +
                        "    \"camera_to_world_columns\": \"right, up, back, position\",\n" +
                        "    \"view_direction\": \"-back\",\n" +
                        "    \"rotation_authority\": \"quaternion and basis vectors; yaw/pitch are convenience fields\"\n" +
                        "  },\n  \"flashback_export\": %s\n}\n",
                JsonText.escape(sessionId), requestedMode.name().toLowerCase(Locale.ROOT), JsonText.escape(csvName), exportDetails);
        Files.writeString(metadata, json, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
    }

    public static void captureCurrentCamera(String source) {
        synchronized (LOCK) {
            if (mode == Mode.OFF || writer == null) {
                return;
            }

            Minecraft client = Minecraft.getInstance();
            Camera camera = client.gameRenderer.getMainCamera();
            if (!camera.isInitialized()) {
                return;
            }

            Vec3 position = camera.position();
            Quaternionf rotation = new Quaternionf(camera.rotation());
            Vector3fc left = camera.leftVector();
            Vector3fc up = camera.upVector();
            Vector3fc forward = camera.forwardVector();
            int width = mode == Mode.FLASHBACK_EXPORT && exportWidth > 0
                    ? exportWidth : client.getWindow().getWidth();
            int height = mode == Mode.FLASHBACK_EXPORT && exportHeight > 0
                    ? exportHeight : client.getWindow().getHeight();
            String dimension = client.level == null ? "unknown" : client.level.dimension().identifier().toString();

            // c2w basis columns: right=-left, up=up, back=-forward.
            String row = String.format(Locale.ROOT,
                    "1,%s,%s,%d,%d," +
                            "%.9f,%.9f,%.9f,%.7f,%.7f," +
                            "%.9f,%.9f,%.9f,%.9f," +
                            "%.9f,%.9f,%.9f," +
                            "%.9f,%.9f,%.9f," +
                            "%.9f,%.9f,%.9f," +
                            "%.7f,%d,%d,%s,%s",
                    sessionId, mode.name().toLowerCase(Locale.ROOT), frameIndex, System.nanoTime(),
                    position.x, position.y, position.z, camera.yaw(), camera.xRot(),
                    rotation.x, rotation.y, rotation.z, rotation.w,
                    -left.x(), -left.y(), -left.z(),
                    up.x(), up.y(), up.z(),
                    -forward.x(), -forward.y(), -forward.z(),
                    camera.getFov(), width, height, dimension, source);
            try {
                writer.write(row);
                writer.newLine();
                if ((frameIndex & 31) == 0) {
                    writer.flush();
                }
                frameIndex++;
            } catch (IOException error) {
                System.err.println("[V2DiT Camera Logger] Write failed: " + error.getMessage());
                stopLocked();
            }
        }
    }

    public static void stop() {
        synchronized (LOCK) {
            stopLocked();
        }
    }

    private static void stopLocked() {
        if (writer != null) {
            try {
                writer.flush();
                writer.close();
            } catch (IOException error) {
                System.err.println("[V2DiT Camera Logger] Close failed: " + error.getMessage());
            }
        }
        writer = null;
        mode = Mode.OFF;
        frameIndex = 0;
        exportWidth = 0;
        exportHeight = 0;
    }
}
