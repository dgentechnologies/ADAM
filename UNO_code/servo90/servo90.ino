#include <Servo.h>

// Create servo objects to control the servos
Servo servo1;  
Servo servo2;  

void setup() {
  // Attach the servos to their respective pins
  servo1.attach(9);  
  servo2.attach(10);

  // Move both servos to the 90-degree (middle) position
  servo1.write(90);
  servo2.write(90);
}

void loop() {
  // The loop is left empty because we only need to set the angle once
}