package io.v2dit.camera.mixin;

import com.moulberry.flashback.exporting.ExportJob;
import com.moulberry.flashback.exporting.ExportSettings;
import com.moulberry.flashback.exporting.SaveableFramebufferQueue;
import com.moulberry.flashback.exporting.VideoWriter;
import io.v2dit.camera.CameraCsvRecorder;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(value = ExportJob.class, remap = false)
public abstract class FlashbackExportJobMixin {
    @Shadow @Final
    private ExportSettings settings;

    @Inject(method = "doExport", at = @At("HEAD"))
    private void v2dit$startTrajectory(VideoWriter writer, SaveableFramebufferQueue queue, CallbackInfo ci) {
        CameraCsvRecorder.startFlashbackExport(this.settings);
    }

    @Inject(method = "doExport", at = @At("TAIL"))
    private void v2dit$stopTrajectory(VideoWriter writer, SaveableFramebufferQueue queue, CallbackInfo ci) {
        CameraCsvRecorder.stop();
    }

    @Inject(
            method = "doExport",
            at = @At(
                    value = "INVOKE",
                    target = "Lcom/moulberry/flashback/exporting/SaveableFramebufferQueue;startDownload(Lcom/mojang/blaze3d/pipeline/RenderTarget;Lcom/moulberry/flashback/exporting/SaveableFramebuffer;Z)V"
            )
    )
    private void v2dit$captureEncodedFrame(VideoWriter writer, SaveableFramebufferQueue queue, CallbackInfo ci) {
        if (CameraCsvRecorder.isExportRecording()) {
            CameraCsvRecorder.captureCurrentCamera("flashback_export");
        }
    }
}
