#include <ros.h>
#include <std_msgs/Int64.h>

ros::NodeHandle nh;
ros::Publisher chatter("result", &res_msg);
void messageCb( const std_msgs::Int64& a){ 

int res;
res = a.data+2;
res_msg.data=res;
chatter.publish( &res_msg);
}

std_msgs::Int64 res_msg;

ros::Subscriber<std_msgs::Int64> sub("to_add", &messageCb);

void setup() {
nh.initNode();
nh.advertise(chatter); 
}

void loop() {
nh.spinOnce(); 
delay(1);
}
