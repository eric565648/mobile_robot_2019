#include <ros.h>
#include <std_msgs/Int64.h>
char cmd;
boolean run = true;

const byte LEFT1 = 13;
const byte LEFT2 = 12;
const byte LEFT_PWM = 11;

const byte RIGHT1 = 8;
const byte RIGHT2 = 9;
const byte RIGHT_PWM = 10;

byte L_motorSpeed = 0;
byte R_motorSpeed = 0;

ros::NodeHandle nh;
void pwmLCb( const std_msgs::Int64& msg){
 
  int l_speed;
  l_speed = msg.data;
  if(l_speed < -255 || l_speed > 255)
    return;
  
  if(l_speed < 0){
    digitalWrite(LEFT1, LOW);
    digitalWrite(LEFT2, HIGH);
    
    L_motorSpeed = -1*l_speed;
  }
  else{
    digitalWrite(LEFT1, HIGH);
    digitalWrite(LEFT2, LOW);
    
    L_motorSpeed = l_speed;
  }
}
void pwmRCb( const std_msgs::Int64& msg){ 
  int r_speed;
  r_speed = msg.data;
  if(r_speed < -255 || r_speed > 255)
    return;
  
  if(r_speed < 0){
    digitalWrite(RIGHT1, LOW);
    digitalWrite(RIGHT2, HIGH);
    
    R_motorSpeed = -1*r_speed;
  }
  else{
    digitalWrite(RIGHT1, HIGH);
    digitalWrite(RIGHT2, LOW);
    
    R_motorSpeed = r_speed;
  }
}
ros::Subscriber<std_msgs::Int64> sub_L("/pwm_L", &pwmLCb);
ros::Subscriber<std_msgs::Int64> sub_R("/pwm_R", &pwmRCb);

void forward() {
 digitalWrite(LEFT1, HIGH);
 digitalWrite(LEFT2, LOW);
 digitalWrite(RIGHT1, HIGH);
 digitalWrite(RIGHT2, LOW);
}

void backward() {
 digitalWrite(LEFT1, LOW);
 digitalWrite(LEFT2, HIGH);
 digitalWrite(RIGHT1, LOW);
 digitalWrite(RIGHT2, HIGH);
}

void turnLeft() {
 digitalWrite(LEFT1, LOW);
 digitalWrite(LEFT2, HIGH);
 digitalWrite(RIGHT1, HIGH);
 digitalWrite(RIGHT2, LOW);
}

void turnRight() {
 digitalWrite(LEFT1, HIGH);
 digitalWrite(LEFT2, LOW);
 digitalWrite(RIGHT1, LOW);
 digitalWrite(RIGHT2, HIGH);
}

void setup() {
  nh.initNode();
  pinMode(LEFT1, OUTPUT);
  pinMode(LEFT2, OUTPUT);
  pinMode(LEFT_PWM, OUTPUT);
  pinMode(RIGHT1, OUTPUT);
  pinMode(RIGHT2, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT);
}

void loop() {
  nh.spinOnce(); 
  
  if (run) {
    analogWrite(LEFT_PWM, L_motorSpeed);
    analogWrite(RIGHT_PWM, R_motorSpeed);
  } else {
    analogWrite(LEFT_PWM, 0);
    analogWrite(RIGHT_PWM, 0);
  }
  
  delay(1);
}
  
