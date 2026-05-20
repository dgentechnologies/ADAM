// ADAM v30 - ESP32-CAM Wi-Fi Sensor & Vision Node
// Serves MJPEG Camera Stream, LiDAR distances, Touch states, and controls Tilt Servo.

#include "esp_camera.h"
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>
#include <ESP32Servo.h>
#include "esp_http_server.h"

// ==========================================
// 1. WIFI CREDENTIALS (UPDATE THESE)
// ==========================================
const char* ssid = "Airtel___DGEN";
const char* password = "GolaKeteDebo#420";

// ==========================================
// 2. PIN ASSIGNMENTS (Maximized for ESP32-CAM)
// ==========================================
#define I2C_SDA     14
#define I2C_SCL     15
#define TILT_PIN    13
#define TOUCH1      2
#define TOUCH2      12
#define TOUCH3_RX   3   // Used as RX during boot, then becomes Touch 3
#define TOUCH4_TX   1   // Used as TX during boot, then becomes Touch 4

// Camera pins (Standard AI Thinker Module)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==========================================
// GLOBALS
// ==========================================
Adafruit_VL53L0X lox = Adafruit_VL53L0X();
bool lidar_active = false;
Servo tiltServo;
httpd_handle_t server_httpd = NULL;

// ==========================================
// HTTP HANDLERS
// ==========================================

// GET /sensors -> Returns JSON with touch states and LiDAR distance
static esp_err_t sensors_handler(httpd_req_t *req) {
    char json_response[128];
    int t1 = digitalRead(TOUCH1);
    int t2 = digitalRead(TOUCH2);
    int t3 = digitalRead(TOUCH3_RX);
    int t4 = digitalRead(TOUCH4_TX);
    
    int dist = -1;
    if (lidar_active) {
        VL53L0X_RangingMeasurementData_t measure;
        lox.rangingTest(&measure, false);
        if(measure.RangeStatus != 4) {  // 4 means out of range
            dist = measure.RangeMilliMeter;
        }
    }

    snprintf(json_response, sizeof(json_response), 
             "{\"touch\":[%d,%d,%d,%d], \"lidar_mm\":%d}", 
             t1, t2, t3, t4, dist);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, strlen(json_response));
}

// GET /tilt?angle=90 -> Moves the MG90S servo
static esp_err_t tilt_handler(httpd_req_t *req) {
    char buf[64];
    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        char param[16];
        if (httpd_query_key_value(buf, "angle", param, sizeof(param)) == ESP_OK) {
            int angle = atoi(param);
            // Constrain angle to safe limits for the neck mechanism
            angle = constrain(angle, 50, 120);
            tiltServo.write(angle);
            return httpd_resp_send(req, "OK", 2);
        }
    }
    httpd_resp_send_404(req);
    return ESP_OK;
}

// GET /stream -> Delivers MJPEG Video Stream to Raspberry Pi
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    while(true){
        fb = esp_camera_fb_get();
        if (!fb) {
            res = ESP_FAIL;
            break;
        }
        
        _jpg_buf_len = fb->len;
        _jpg_buf = fb->buf;

        if (res == ESP_OK) res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        }
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        
        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;
        
        // Brief delay to allow sensors/tilt HTTP requests to be processed
        vTaskDelay(pdMS_TO_TICKS(20)); 
    }
    return res;
}

// ==========================================
// INITIALIZATION
// ==========================================
void startServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.max_uri_handlers = 8; // Allow multiple endpoints

    httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
    httpd_uri_t sensors_uri = { .uri = "/sensors", .method = HTTP_GET, .handler = sensors_handler, .user_ctx = NULL };
    httpd_uri_t tilt_uri = { .uri = "/tilt", .method = HTTP_GET, .handler = tilt_handler, .user_ctx = NULL };

    if (httpd_start(&server_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(server_httpd, &stream_uri);
        httpd_register_uri_handler(server_httpd, &sensors_uri);
        httpd_register_uri_handler(server_httpd, &tilt_uri);
    }
}

void setup() {
    // 1. BOOT SERIAL (To show IP address)
    Serial.begin(115200);
    delay(500);

    // 2. INIT CAMERA
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_VGA; // 640x480 for smooth Pi CV
    config.jpeg_quality = 12;
    config.fb_count = 2; // Double buffering requires PSRAM

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed!");
    }

    // 3. INIT I2C & LIDAR
    Wire.begin(I2C_SDA, I2C_SCL);
    if (!lox.begin()) {
        Serial.println("Failed to boot VL53L0X LiDAR");
    } else {
        lidar_active = true;
        Serial.println("VL53L0X LiDAR Ready");
    }

    // 4. INIT SERVO
    ESP32PWM::allocateTimer(0);
    tiltServo.setPeriodHertz(50);
    tiltServo.attach(TILT_PIN, 500, 2400); 
    tiltServo.write(85); // Center position

    // 5. INIT TOUCH 1 & 2
    pinMode(TOUCH1, INPUT);
    pinMode(TOUCH2, INPUT);

    // 6. CONNECT WIFI
    WiFi.setHostname("ADAM-EYES");
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");
    Serial.print("TARGET IP ADDRESS: http://");
    Serial.println(WiFi.localIP());

    startServer();
    
    // 7. FREE UP TX/RX FOR TOUCH SENSORS
    Serial.println("\n!!! WARNING !!!");
    Serial.println("Disabling Serial now to use TX/RX pins for Touch 3 & 4.");
    Serial.println("You will no longer see Serial output. Setup complete!");
    delay(4000); // Give you 4 seconds to read the IP address
    
    Serial.flush();
    Serial.end(); // Disconnects UART driver

    // Now it's safe to use TX/RX as standard digital inputs
    pinMode(TOUCH3_RX, INPUT);
    pinMode(TOUCH4_TX, INPUT);
}

void loop() {
    // HTTP Server runs asynchronously. Just yield to background tasks.
    delay(10);
}