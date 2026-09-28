// UART receiver for ESP32. Pin numbers must match your board and wiring.
// Drive an active buzzer through a transistor/MOSFET, never directly from GPIO.
constexpr int RX_PIN = 16;
constexpr int TX_PIN = 17;
constexpr int LED_PIN = 2;
constexpr int BUZZER_PIN = 25;
constexpr unsigned long WATCHDOG_MS = 3000;
constexpr unsigned long MAX_SOUND_MS = 10000;
bool alarm = false, fault = true;
unsigned long lastHeartbeat = 0, alarmSince = 0;
String command;

void processCommand(const String &line) {
  if (line == "ALARM:1" || line == "ALARM:0") {
    bool next = line == "ALARM:1";
    if (next && !alarm) alarmSince = millis();
    alarm = next;
    lastHeartbeat = millis();
  } else if (line == "FAULT:1" || line == "FAULT:0") {
    fault = line == "FAULT:1";
    lastHeartbeat = millis();
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial2.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);
  command.reserve(32);
}

void loop() {
  while (Serial2.available()) {
    char c = Serial2.read();
    if (c == '\n') { processCommand(command); command = ""; }
    else if (c != '\r') {
      command += c;
      if (command.length() > 31) command = "";
    }
  }
  unsigned long now = millis();
  if (now - lastHeartbeat > WATCHDOG_MS) { alarm = false; fault = true; }
  bool sound = alarm && !fault && now - alarmSince < MAX_SOUND_MS;
  digitalWrite(BUZZER_PIN, sound ? HIGH : LOW);
  digitalWrite(LED_PIN, fault ? ((now / 500) % 2) : alarm);
}
