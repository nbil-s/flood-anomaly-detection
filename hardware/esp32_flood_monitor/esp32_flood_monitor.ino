
#include <DHT.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
#include "secrets.h"

#define DHT_PIN 17
#define DHT_TYPE DHT22

#define WATER_LEVEL_PIN 34
#define RAIN_SENSOR_PIN 35

const char* WIFI_SSID = "...";
const char* WIFI_PASSWORD = "...";

const char* SERVER_URL =
  "http://192.168.100.133:8000/physical-readings";

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);
  dht.begin();

  Serial.println("Flood monitoring system starting...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected.");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  // Synchronise time for valid UTC timestamps.
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");

  Serial.print("Synchronising time");

  while (time(nullptr) < 1700000000) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nTime synchronised.");
}

void loop() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  int waterLevel = analogRead(WATER_LEVEL_PIN);
  int rainfall = analogRead(RAIN_SENSOR_PIN);

  Serial.println("\n------------------------------");

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT22 reading failed. Skipping POST.");
    delay(5000);
    return;
  }

  Serial.print("Temperature: ");
  Serial.println(temperature);

  Serial.print("Humidity: ");
  Serial.println(humidity);

  Serial.print("Water level raw: ");
  Serial.println(waterLevel);

  Serial.print("Rain sensor raw: ");
  Serial.println(rainfall);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    time_t now = time(nullptr);

    struct tm timeInfo;
    gmtime_r(&now, &timeInfo);

    char timestamp[30];

    strftime(
      timestamp,
      sizeof(timestamp),
      "%Y-%m-%dT%H:%M:%SZ",
      &timeInfo
    );

    String payload = "{";
    payload += "\"timestamp\":\"" + String(timestamp) + "\",";
    payload += "\"water_level_raw\":" + String(waterLevel) + ",";
    payload += "\"rainfall_raw\":" + String(rainfall) + ",";
    payload += "\"temperature\":" + String(temperature, 2) + ",";
    payload += "\"humidity\":" + String(humidity, 2);
    payload += "}";

    Serial.println("Sending readings to backend...");

    int httpCode = http.POST(payload);

    Serial.print("Backend HTTP response: ");
    Serial.println(httpCode);

    if (httpCode > 0) {
      Serial.println("Backend response:");
      Serial.println(http.getString());
    } else {
      Serial.println("HTTP request failed.");
    }

    http.end();

  } else {
    Serial.println("WiFi disconnected. Reconnecting...");
    WiFi.reconnect();
  }

  delay(30000);
}
