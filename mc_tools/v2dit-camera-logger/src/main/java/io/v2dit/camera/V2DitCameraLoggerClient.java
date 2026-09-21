package io.v2dit.camera;

import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import org.lwjgl.glfw.GLFW;

public final class V2DitCameraLoggerClient implements ClientModInitializer {
    private static KeyMapping toggleManual;

    @Override
    public void onInitializeClient() {
        toggleManual = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.v2dit_camera_logger.toggle_manual",
                InputConstants.Type.KEYSYM,
                GLFW.GLFW_KEY_F8,
                KeyMapping.Category.MISC
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleManual.consumeClick()) {
                if (CameraCsvRecorder.isManualRecording()) {
                    CameraCsvRecorder.stop();
                    notifyPlayer(client, "V2DiT 相机轨迹记录已停止");
                } else if (CameraCsvRecorder.isExportRecording()) {
                    notifyPlayer(client, "Flashback 正在自动导出轨迹，无需手动开始");
                } else if (CameraCsvRecorder.startManual()) {
                    notifyPlayer(client, "V2DiT 相机轨迹记录已开始（F8 停止）");
                }
            }
        });

        LevelRenderEvents.END_MAIN.register(context -> {
            if (CameraCsvRecorder.isManualRecording()) {
                CameraCsvRecorder.captureCurrentCamera("manual_render");
            }
        });
    }

    private static void notifyPlayer(Minecraft client, String message) {
        client.gui.setOverlayMessage(Component.literal(message), false);
    }
}
