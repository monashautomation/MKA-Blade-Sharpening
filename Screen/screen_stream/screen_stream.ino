/**
 * Low-latency MJPEG viewer for the Waveshare ESP32-S3-Touch-LCD-7 (800x480 RGB).
 *
 * Pulls frames from the dedicated pull-based feed served by ../picam_2/app.py
 * (serve_lcd_stream, default port 8080) and decodes each JPEG with JPEGDEC into
 * the RGB framebuffer - no LVGL in the video path.
 *
 * Latency choices:
 *   - pull protocol: request one frame, receive the freshest -> nothing queues
 *     in TCP buffers, so the panel never falls behind (kills the ~1s lag)
 *   - WiFi.setSleep(false) so the modem never naps between frames
 *   - double-buffered: decode into the off-screen framebuffer, then swap it in
 *     at VSYNC via LCD::switchFrameBufferTo -> tear-free, no extra copy
 *   - only present a fully-decoded frame (never a partial/corrupt one)
 *
 * Board-config note (for high frame rate over the RGB bus + PSRAM), set in
 * `sdkconfig` / menuconfig, mirrors the values that make LVGL smooth here:
 *   CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
 *   CONFIG_SPIRAM_MODE_OCT=y, CONFIG_SPIRAM_SPEED_120M=y (with 120M flash)
 *   CONFIG_SPIRAM_FETCH_INSTRUCTIONS=y, CONFIG_SPIRAM_RODATA=y
 *   CONFIG_COMPILER_OPTIMIZATION_PERF=y
 * Arduino IDE: Board = ESP32S3 Dev Module, PSRAM = "OPI PSRAM",
 *              Flash Size = 16MB, Partition = "Huge APP".
 *
 * Requires libraries: ESP32_Display_Panel (>=1.0.4), JPEGDEC (bitbank2).
 * The board is selected in esp_panel_board_supported_conf.h
 * (BOARD_WAVESHARE_ESP32_S3_TOUCH_LCD_7), same as the simple_rotation example.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_display_panel.hpp>
#include <JPEGDEC.h>

using namespace esp_panel::drivers;
using namespace esp_panel::board;

// ---------------------------------------------------------------------------
// User configuration - EDIT THESE
// ---------------------------------------------------------------------------
#define WIFI_SSID    "PiCamAP"
#define WIFI_PASS    "picamstream"
#define STREAM_HOST  "192.168.4.1"   // IP printed by app.py ("ESP32-S3 panel feed")
#define STREAM_PORT  8080             // config.py -> LCD_STREAM_PORT

// Max size of a single JPEG frame. 640x480 q80 stays well under this.
#define JPEG_BUF_SIZE (128 * 1024)

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static Board    *board  = nullptr;
static LCD      *lcd    = nullptr;
static JPEGDEC   jpeg;
static WiFiClient client;

static uint8_t *jpegBuf  = nullptr;        // raw JPEG bytes off the wire
static uint16_t *fb[2]   = { nullptr, nullptr };  // the panel's two framebuffers
static int      backIdx  = 0;              // buffer we decode into next (off-screen)
static uint16_t *curFB   = nullptr;        // == fb[backIdx] during a decode
static int      LCD_W    = 800;
static int      LCD_H    = 480;
static int      off_x    = 0;              // top-left where the frame is placed
static int      off_y    = 0;
static volatile bool s_vsync = false;      // set at each panel refresh-finish (VSYNC)

// Fires at the end of every panel refresh (start of vertical blanking).
static bool IRAM_ATTR onRefreshFinish(void *user)
{
    s_vsync = true;
    return false;
}

// Block until the next refresh finishes (i.e. we are in the blanking interval),
// or until timeout so we can never hang.
static void waitForVsync(uint32_t timeout_ms)
{
    s_vsync = false;
    uint32_t t0 = millis();
    while (!s_vsync && (millis() - t0) < timeout_ms) delay(1);
}

// ---------------------------------------------------------------------------
// JPEGDEC -> panel: called once per decoded MCU block
// ---------------------------------------------------------------------------
static int jpegDrawCallback(JPEGDRAW *p)
{
    // Write this MCU block into the OFF-screen framebuffer (stride = LCD_W).
    // The panel is still scanning the other buffer, so nothing tears here.
    for (int row = 0; row < p->iHeight; row++) {
        int y = off_y + p->y + row;
        if (y < 0 || y >= LCD_H) continue;
        int x = off_x + p->x;
        if (x < 0) continue;
        int w = p->iWidth;
        if (x + w > LCD_W) w = LCD_W - x;       // clip right edge if oversized
        if (w <= 0) continue;
        memcpy(&curFB[y * LCD_W + x],
               &p->pPixels[row * p->iWidth],
               (size_t)w * 2);
    }
    return 1;   // keep decoding
}

// ---------------------------------------------------------------------------
// On-screen FPS overlay: a tiny 3x5 digit font stamped into the framebuffer
// ---------------------------------------------------------------------------
static uint32_t fpsCount = 0;
static uint32_t fpsT0    = 0;
static int      fpsValue = 0;

static const uint8_t DIGIT3x5[10][5] = {
    {0b111,0b101,0b101,0b101,0b111}, {0b010,0b110,0b010,0b010,0b111},
    {0b111,0b001,0b111,0b100,0b111}, {0b111,0b001,0b111,0b001,0b111},
    {0b101,0b101,0b111,0b001,0b001}, {0b111,0b100,0b111,0b001,0b111},
    {0b111,0b100,0b111,0b101,0b111}, {0b111,0b001,0b010,0b010,0b010},
    {0b111,0b101,0b111,0b101,0b111}, {0b111,0b101,0b111,0b001,0b111},
};

static void fbFillRect(int x, int y, int w, int h, uint16_t color)
{
    for (int yy = y; yy < y + h; yy++) {
        if (yy < 0 || yy >= LCD_H) continue;
        for (int xx = x; xx < x + w; xx++) {
            if (xx < 0 || xx >= LCD_W) continue;
            curFB[yy * LCD_W + xx] = color;
        }
    }
}

// Stamp the fps number (white on a black box) into the top-left of curFB.
static void drawFpsOverlay(int fps)
{
    if (fps < 0)   fps = 0;
    if (fps > 999) fps = 999;
    char buf[8];
    int n = snprintf(buf, sizeof(buf), "%d", fps);

    const int s = 5, dw = 3 * s, gap = s, m = 4;   // scale, digit width, gap, margin
    const int x0 = 6, y0 = 6;
    fbFillRect(x0, y0, n * dw + (n - 1) * gap + 2 * m, 5 * s + 2 * m, 0x0000);

    int x = x0 + m, y = y0 + m;
    for (int i = 0; i < n; i++) {
        const uint8_t *g = DIGIT3x5[buf[i] - '0'];
        for (int row = 0; row < 5; row++)
            for (int col = 0; col < 3; col++)
                if (g[row] & (1 << (2 - col)))
                    fbFillRect(x + col * s, y + row * s, s, s, 0xFFFF);
        x += dw + gap;
    }
}

static void drawJpeg(uint8_t *buf, int len)
{
    if (!jpeg.openRAM(buf, len, jpegDrawCallback)) {
        return;
    }
    // RGB565 little-endian matches the panel's framebuffer word order here.
    // If colors look byte-swapped, change this to RGB565_BIG_ENDIAN.
    jpeg.setPixelType(RGB565_LITTLE_ENDIAN);

    int jw = jpeg.getWidth();   if (jw > LCD_W) jw = LCD_W;
    int jh = jpeg.getHeight();  if (jh > LCD_H) jh = LCD_H;
    off_x = (LCD_W - jw) / 2;   if (off_x < 0) off_x = 0;   // center (letterbox)
    off_y = (LCD_H - jh) / 2;   if (off_y < 0) off_y = 0;

    curFB = fb[backIdx];                         // decode into the off-screen buffer
    if (jpeg.decode(0, 0, 0)) {
        drawFpsOverlay(fpsValue);                // stamp the last measured fps onto the frame
        // 1) Wait for the panel to finish scanning the current frame, so we are
        //    inside the vertical blanking interval (nothing is being drawn).
        waitForVsync(100);
        // 2) Switch now, during blanking -> the whole next frame is scanned from
        //    the new buffer, so the switch never lands mid-scanline.
        lcd->switchFrameBufferTo(fb[backIdx]);
        // 3) Let that swap take effect before we reuse the other buffer, so we
        //    never decode into a buffer that is still on screen.
        waitForVsync(100);
        backIdx ^= 1;                            // other buffer is now off-screen -> safe

        // Measure displayed frame rate; refresh the on-screen number once a second.
        fpsCount++;
        uint32_t now = millis();
        if (now - fpsT0 >= 1000) {
            fpsValue = (int)(fpsCount * 1000 / (now - fpsT0));
            fpsCount = 0;
            fpsT0 = now;
            Serial.printf("FPS: %d\n", fpsValue);
        }
    }
    jpeg.close();
}

// ---------------------------------------------------------------------------
// Pull-based frame protocol over a raw WiFiClient
//
// Wire format, one round per frame:
//   ESP -> Pi : 1 request byte
//   Pi  -> ESP: 4-byte big-endian length, then that many JPEG bytes (freshest)
// One frame in flight at a time -> nothing queues -> no lag buildup.
// ---------------------------------------------------------------------------

// Read exactly len bytes into buf. Returns false on timeout / disconnect.
static bool readFull(uint8_t *buf, int len, uint32_t timeout_ms)
{
    int got = 0;
    uint32_t start = millis();
    while (got < len) {
        int n = client.read(buf + got, len - got);
        if (n > 0) {
            got += n;
            start = millis();
        } else {
            if (!client.connected() && client.available() == 0) return false;
            if (millis() - start > timeout_ms) return false;
            delay(1);
        }
    }
    return true;
}

// Connect, then request/receive/decode one freshest frame at a time.
static void streamLoop()
{
    Serial.printf("Connecting to %s:%d ...\n", STREAM_HOST, STREAM_PORT);
    if (!client.connect(STREAM_HOST, STREAM_PORT)) {
        Serial.println("  connect failed");
        return;
    }
    client.setNoDelay(true);
    Serial.println("Streaming (pull)...");

    while (client.connected()) {
        // Ask for the next (freshest) frame.
        const uint8_t req = 1;
        if (client.write(&req, 1) != 1) break;

        // Read the 4-byte big-endian length header.
        uint8_t hdr[4];
        if (!readFull(hdr, 4, 5000)) break;
        uint32_t len = ((uint32_t)hdr[0] << 24) | ((uint32_t)hdr[1] << 16) |
                       ((uint32_t)hdr[2] << 8)  |  (uint32_t)hdr[3];
        if (len == 0 || len > JPEG_BUF_SIZE) break;

        if (!readFull(jpegBuf, len, 5000)) break;
        drawJpeg(jpegBuf, len);
    }

    client.stop();
    Serial.println("Stream ended.");
}

// ---------------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------------
void setup()
{
    Serial.begin(115200);

    Serial.println("Initializing board");
    board = new Board();
    board->init();
    lcd = board->getLCD();
    lcd->configFrameBufferNumber(2);          // double-buffering -> tear-free
    if (!board->begin()) {
        Serial.println("Board begin failed!");
        while (true) delay(1000);
    }
    LCD_W = lcd->getFrameWidth();
    LCD_H = lcd->getFrameHeight();
    Serial.printf("Panel %dx%d\n", LCD_W, LCD_H);

    fb[0]   = (uint16_t *)lcd->getFrameBufferByIndex(0);
    fb[1]   = (uint16_t *)lcd->getFrameBufferByIndex(1);
    jpegBuf = (uint8_t *)ps_malloc(JPEG_BUF_SIZE);
    if (!fb[0] || !fb[1] || !jpegBuf) {
        Serial.println("Framebuffer/PSRAM alloc failed!");
        while (true) delay(1000);
    }
    // Clear both buffers once; the letterbox bars are never written, so they
    // stay black, and each frame fully overwrites the centered image region.
    memset(fb[0], 0, (size_t)LCD_W * LCD_H * 2);
    memset(fb[1], 0, (size_t)LCD_W * LCD_H * 2);

    lcd->attachRefreshFinishCallback(onRefreshFinish);   // VSYNC signal for the swap

    Serial.printf("Connecting to WiFi \"%s\"", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        delay(300);
        Serial.print(".");
    }
    WiFi.setSleep(false);   // no modem sleep -> lower, steadier latency
    Serial.printf("\nWiFi up, IP %s\n", WiFi.localIP().toString().c_str());
}

void loop()
{
    if (WiFi.status() != WL_CONNECTED) {
        WiFi.reconnect();
        delay(500);
        return;
    }
    streamLoop();     // blocks while streaming; returns on error/disconnect
    delay(200);       // brief backoff before reconnecting
}
