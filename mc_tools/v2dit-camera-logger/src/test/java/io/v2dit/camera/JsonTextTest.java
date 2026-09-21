package io.v2dit.camera;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class JsonTextTest {
    @Test
    void acceptsUnnamedFlashbackExportJobs() {
        assertEquals("", JsonText.escape(null));
    }

    @Test
    void escapesJsonControlCharacters() {
        assertEquals("a\\\\b\\\"c\\r\\nd\\te", JsonText.escape("a\\b\"c\r\nd\te"));
    }
}
