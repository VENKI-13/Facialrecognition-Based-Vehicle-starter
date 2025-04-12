int ledPin = 9;     // Pin connected to LED
int buttonPin = 7;  // Pin connected to the external button
int motorPin = 11;  // Pin connected to relay (motor)
int ledState = LOW; // Initially, the LED is off
int motorState = LOW; // Initially, the motor is off
int buttonState;
int lastButtonState = HIGH; // For detecting button press
bool cameraTrigger = false;

void setup() {
    pinMode(ledPin, OUTPUT);
    pinMode(motorPin, OUTPUT);
    pinMode(buttonPin, INPUT_PULLUP);
    Serial.begin(9600);

    // Ensure motor and LED are off at the start
    digitalWrite(ledPin, LOW);
    digitalWrite(motorPin, LOW);
}

void loop() {
    buttonState = digitalRead(buttonPin);

    // Detect button press (falling edge detection)
    if (buttonState == LOW && lastButtonState == HIGH) {
        cameraTrigger = !cameraTrigger;  // Toggle camera state

        if (cameraTrigger) {
            Serial.println("open_camera"); // Notify Python to start camera
        } else {
            Serial.println("stop_led");   // Notify Python to turn off LED
            digitalWrite(ledPin, LOW);   // Turn off LED
            digitalWrite(motorPin, LOW); // Turn off motor
        }
        
        delay(300); // Debounce delay
    }

    // Check for serial command from Python
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');

        if (command == "start") {
            ledState = HIGH; 
            motorState = HIGH;
            digitalWrite(ledPin, ledState);
            digitalWrite(motorPin, motorState); // Turn ON relay (motor)
        } else if (command == "stop") {
            ledState = LOW;
            motorState = LOW;
            digitalWrite(ledPin, ledState);
            digitalWrite(motorPin, motorState); // Turn OFF relay (motor)
        }
    }

    lastButtonState = buttonState; // Update last button state
}
