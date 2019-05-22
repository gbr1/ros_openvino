/* 
 * This file is part of the ros_openvino package (https://github.com/gbr1/ros_openvino or http://gbr1.github.io).
 * Copyright (c) 2019 Giovanni di Dio Bruno.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <ros/ros.h>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <ros_openvino/Object.h>
#include <ros_openvino/Objects.h>
#include <ros_openvino/ObjectBox.h>
#include <ros_openvino/ObjectBoxList.h>
#include <string>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sstream>


using namespace InferenceEngine;

//Frames required in object detection with async api
cv::Mat frame_now;  
cv::Mat frame_next;
cv::Mat depth_frame; 

//Parameters for camera calibration
float fx;
float fy;
float cx;
float cy;

//Frames sizes
size_t color_width;
size_t color_height;
size_t depth_width;
size_t depth_height;

//lockers
bool cf_available=false;
bool is_last_frame=true;

//ROS parameters
std::string device;
float confidence_threshold;
std::string network_path;
std::string weights_path;
std::string labels_path;
std::string colors_path;
bool output_as_image;
bool output_as_list;
bool depth_analysis;
bool output_markers;
bool output_markerslabel;
std::string depth_frameid;
float markerduration;
bool output_boxlist;

//ROS messages
sensor_msgs::Image output_image_msg;
ros_openvino::Object tmp_object;
ros_openvino::Objects results_list;
visualization_msgs::Marker marker;
visualization_msgs::Marker marker_label;
visualization_msgs::MarkerArray markers;
ros_openvino::ObjectBox tmp_box;
ros_openvino::ObjectBoxList box_list;


//Couple of hex in string format to uint value (FF->255)
uint8_t hexToUint8(std::string s){
    unsigned int x;
    std::stringstream ss;
    ss<<std::hex<<s;
    ss>>x;
    return x;
}

//OpenCV mat to blob
static InferenceEngine::Blob::Ptr mat_to_blob(const cv::Mat &image) {
    InferenceEngine::TensorDesc tensor(InferenceEngine::Precision::U8,{1, (size_t)image.channels(), (size_t)image.size().height, (size_t)image.size().width},InferenceEngine::Layout::NHWC);
    return InferenceEngine::make_shared_blob<uint8_t>(tensor, image.data);
}

//Image to blob
void frame_to_blob(const cv::Mat& image,InferRequest::Ptr& analysis,const std::string& descriptor) {
    analysis->SetBlob(descriptor, mat_to_blob(image));
}

//Callback: a new RGB image is arrived
void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg){
    cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
    cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
    if(!cf_available){
        color_mat.copyTo(frame_now);
        cf_available=true;
    }
    color_mat.copyTo(frame_next);
    is_last_frame=false;
    color_width  = (size_t)color_mat.size().width;
    color_height = (size_t)color_mat.size().height;
}

//Callaback: a new CameraInfo is arrived
void infoCallback(const sensor_msgs::CameraInfo::ConstPtr& info_msg){
    fx=info_msg->K[0];
    fy=info_msg->K[4];
    cx=info_msg->K[2];
    cy=info_msg->K[5];        
}

//Callback: a new depth image is arrived
void depthCallback(const sensor_msgs::Image::ConstPtr& depth_msg){
    cv::Mat depth_mat(depth_msg->height, depth_msg->width, CV_MAKETYPE(CV_16U,1),const_cast<uchar*>(&depth_msg->data[0]), depth_msg->step);
    depth_mat.copyTo(depth_frame);
    depth_width  = (size_t)depth_mat.size().width;
    depth_height = (size_t)depth_mat.size().height;
}

//Main Function
int main(int argc, char **argv){
    try{
        //Initialize Ros
        ros::init(argc, argv, "object_detection");
        //Handle creation
        ros::NodeHandle n;

        //--------- ROS PARAMETERS ----------//

        //target device, by default is MYRIAD due ai core as mainly supported device
        if (n.getParam("/object_detection/target",device)){
            ROS_INFO("Target: %s", device.c_str());
        }  
        else{
            device="MYRIAD";
            ROS_INFO("[Default] Target: %s", device.c_str());
        }
        
        //threshold for confidence
        if (n.getParam("/object_detection/threshold", confidence_threshold)){
            ROS_INFO("Confidence Threshold: %f", confidence_threshold);
            
        }  
        else{
            confidence_threshold=0.5;
            ROS_INFO("[Default] Confidence Threshold: %f", confidence_threshold);
        }

        //network-> model.xml
        if (n.getParam("/object_detection/model_network",network_path)){
            ROS_INFO("Model Network: %s", network_path.c_str());
        }
        else{
            network_path="/opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.xml";
            ROS_INFO("[Default] Model Network: %s", network_path.c_str());
        }

        //weight -> model.bin
        if (n.getParam("/object_detection/model_weights", weights_path)){
            ROS_INFO("Model Weights: %s", weights_path.c_str());
        }
        else{
            weights_path="/opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.bin";
            ROS_INFO("[Default] Model Weights: %s", weights_path.c_str());
        }

        //labels -> model.labels
        if (n.getParam("/object_detection/model_labels", labels_path)){
            ROS_INFO("Model Labels: %s", labels_path.c_str());
        }
        else{
            labels_path="/opt/intel/computer_vision_sdk/deployment_tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.labels";
            ROS_INFO("[Default] Model Labels: %s", labels_path.c_str());
        }

        //labels -> model.labels
        if (n.getParam("/object_detection/model_colors", colors_path)){
            ROS_INFO("Model Colors: %s", colors_path.c_str());
        }
        else{
            colors_path="";
            ROS_INFO("[Default] Model Colors: %s", colors_path.c_str());
        }

        //check if frame publisher is wanted
        if (n.getParam("/object_detection/output_as_image", output_as_image)){
            ROS_INFO("Publish Analyzed Image Topic: %s", output_as_image ? "true" : "false");
        }
        else{
            output_as_image=true;
            ROS_INFO("[Default] Publish Analyzed Image Topic: %s", output_as_image ? "true" : "false");
        }

        //check if results list publisher is wanted
        if (n.getParam("/object_detection/output_as_list", output_as_list)){
            ROS_INFO("Publish Results List: %s", output_as_list ? "true" : "false");
        }
        else{
            output_as_list=true;
            ROS_INFO("[Default] Publish Results List: %s", output_as_list ? "true" : "false");
        }

        //frame id used as TF reference
        if (n.getParam("/object_detection/frame_id", depth_frameid)){
            ROS_INFO("Frame id used: %s", depth_frameid.c_str());
        }
        else{
            depth_frameid="/camera_link";
            ROS_INFO("[Default] Frame id: %s", depth_frameid.c_str());
        }

        //check if depth analysis is wanted
        if (n.getParam("/object_detection/depth_analysis", depth_analysis)){
            ROS_INFO("Depth Analysis: %s", depth_analysis ? "ENABLED" : "DISABLED");
        }
        else{
            depth_analysis=true;
            ROS_INFO("[Default] Depth Analysis: %s", depth_analysis ? "ENABLED" : "DISABLED");
        }

        if (depth_analysis){
            //check if markers are wanted
            if (n.getParam("/object_detection/output_as_markers", output_markers)){
                ROS_INFO("Output Markers: %s", output_markers ? "true" : "false");
            }
            else{
                output_markers=true;
                ROS_INFO("[Default] Output Markers: %s", output_markers ? "true" : "false");
            }

            //check if markers label are wanted
            if (n.getParam("/object_detection/output_as_markerslabel", output_markerslabel)){
                ROS_INFO("Output Markers Label: %s", output_markerslabel ? "true" : "false");
            }
            else{
                output_markerslabel=true;
                ROS_INFO("[Default] Output Markers Label: %s", output_markerslabel ? "true" : "false");
            }
            
            if (output_markers||output_markerslabel){
                //lifetime of markers
                if (n.getParam("/object_detection/output_markers_lifetime", markerduration)){
                    ROS_INFO("Output Markers Lifetime: %f", markerduration);
                }
                else{
                    markerduration=0.1;
                    ROS_INFO("[Default] Output Markers Lifetime: %f", markerduration);
                }
            }

            //check if output as box list is wanted
            if (n.getParam("/object_detection/output_as_box_list", output_boxlist)){
                ROS_INFO("Output as Box List: %s", output_boxlist ? "true" : "false");
            }
            else{
                output_boxlist=true;
                ROS_INFO("[Default] Output as Box List: %s", output_boxlist ? "true" : "false");
            }

        }

        
        //ROS subscribers
        ros::Subscriber image_sub = n.subscribe("/object_detection/input_image",1,imageCallback);
        ros::Subscriber camerainfo_sub;
        ros::Subscriber depth_sub; 
        
        //ROS publishers
        ros::Publisher image_pub;
        if (output_as_image){
            image_pub = n.advertise<sensor_msgs::Image>("/object_detection/output_image",1);
        }
        
        ros::Publisher result_pub;
        if (output_as_list){
            result_pub = n.advertise<ros_openvino::Objects>("/object_detection/results",1);
        }
        
        //Depth analysis allow subscription and publishing
        ros::Publisher marker_pub;
        ros::Publisher boxlist_pub;
        if (depth_analysis){
            depth_sub = n.subscribe("/object_detection/input_depth",1,depthCallback);
            camerainfo_sub = n.subscribe("/object_detection/camera_info",1,infoCallback);
            if (output_markers||output_markerslabel){
                marker_pub = n.advertise<visualization_msgs::MarkerArray>("/object_detection/markers", 1);
            }
            if (output_boxlist){
                boxlist_pub = n.advertise<ros_openvino::ObjectBoxList>("/object_detection/box_list", 1);
            }
        }
        
        //setup inference engine device, default is MYRIAD, but you can choose also GPU or CPU (CPU is not tested)
        InferencePlugin OpenVino_plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(device);

        //Setup model, weights, labels and colors
        CNNNetReader network_reader;
        network_reader.ReadNetwork(network_path);
        network_reader.ReadWeights(weights_path);
        std::vector<std::string> vector_labels;
        std::ifstream inputFileLabel(labels_path);
        std::copy(std::istream_iterator<std::string>(inputFileLabel),std::istream_iterator<std::string>(),std::back_inserter(vector_labels));
        std::vector<std::string> vector_colors;
        std::ifstream inputFileColor(colors_path);
        std::copy(std::istream_iterator<std::string>(inputFileColor),std::istream_iterator<std::string>(),std::back_inserter(vector_colors));

        //setup input stuffs
        InputsDataMap input_info(network_reader.getNetwork().getInputsInfo());
        
        InputInfo::Ptr& input_data = input_info.begin()->second;
        auto inputName = input_info.begin()->first;
        input_data->setPrecision(Precision::U8);

        input_data->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input_data->getInputData()->setLayout(Layout::NHWC);

        //setup output
        OutputsDataMap output_info(network_reader.getNetwork().getOutputsInfo());
        DataPtr& output_data = output_info.begin()->second;
        auto outputName = output_info.begin()->first;
        const int num_classes = network_reader.getNetwork().getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
        if (vector_labels.size() != num_classes) {
        if (vector_labels.size() == (num_classes - 1))
                vector_labels.insert(vector_labels.begin(), "no-label");
            else
                vector_labels.clear();
        }
        const SizeVector output_dimension = output_data->getTensorDesc().getDims();
        const int results_number = output_dimension[2];
        const int object_size = output_dimension[3];
        if ((object_size != 7 || output_dimension.size() != 4)) {
            ROS_ERROR("There is a problem with output dimension");
        }

        output_data->setPrecision(Precision::FP32);
        output_data->setLayout(Layout::NCHW);

        //load model into plugin
        ExecutableNetwork model_network = OpenVino_plugin.LoadNetwork(network_reader.getNetwork(), {});

        //inference request to engine
        InferRequest::Ptr engine_next = model_network.CreateInferRequestPtr();
        InferRequest::Ptr engine_curr = model_network.CreateInferRequestPtr();

        //start stuffs
        bool is_first_frame = true;
        int markers_size=0;
        
        //loop while roscore is up
        while(ros::ok()){
            //call roscore
            ros::spinOnce();

            //if there is a frame
            if (cf_available){ 
                //if first frame is available    
                if (is_first_frame) {
                    frame_to_blob(frame_now,engine_curr, inputName);
                }

                //if there are other frames
                if (!is_last_frame) {
                    frame_to_blob(frame_next, engine_next, inputName);
                } 

                if (is_first_frame) {
                    engine_curr->StartAsync();
                }
                if (!is_last_frame) {
                    engine_next->StartAsync();
                }

                //set to 0 size of markers and delete entire markersarray        
                int kmarker=0;
                markers.markers.clear();

                //if engine give us something
                if (OK == engine_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                    //get results stuffs
                    const float *compute_results = engine_curr->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();  
                    //for all results
                    for (int i = 0; i < results_number; i++) {
                        float result_id = compute_results[i * object_size + 0];
                        int result_label = static_cast<int>(compute_results[i * object_size + 1]);
                        //get confidence
                        float result_confidence = compute_results[i * object_size + 2];
                        //get 2d box
                        float result_xmin = compute_results[i * object_size + 3];
                        float result_ymin = compute_results[i * object_size + 4];
                        float result_xmax = compute_results[i * object_size + 5];
                        float result_ymax = compute_results[i * object_size + 6];  
                        //get label and color
                        std::string text= (result_label < vector_labels.size() ? vector_labels[result_label] : std::string("unknown ") + std::to_string(result_label));
                        std::string color=(result_label < vector_colors.size() ? vector_colors[result_label] : std::string("00FF00"));
                        //rgb color
                        uint8_t colorR = hexToUint8(color.substr(0,2));
                        uint8_t colorG = hexToUint8(color.substr(2,2));
                        uint8_t colorB = hexToUint8(color.substr(4,2));

                        //improve 2d box
                        result_xmin=result_xmin<0.0? 0.0 : result_xmin;
                        result_xmax=result_xmax>1.0? 1.0 : result_xmax;
                        result_ymin=result_ymin<0.0? 0.0 : result_ymin;
                        result_ymax=result_ymax>1.0? 1.0 : result_ymax;   

                        //check threshold
                        if (result_confidence > confidence_threshold){
                            //result topic
                            if (output_as_list){
                                tmp_object.label=text;
                                tmp_object.confidence=result_confidence;
                                tmp_object.x=result_xmin;
                                tmp_object.y=result_ymin;
                                tmp_object.width=result_xmax-result_xmin;
                                tmp_object.height=result_ymax-result_ymin;
                                results_list.objects.push_back(tmp_object);
                            }

                            //if depth analysis for 3d world detection is active
                            if (depth_analysis){

                                cv::Mat subdepthP = depth_frame(cv::Rect(result_xmin*depth_width,result_ymin*depth_height,(result_xmax-result_xmin)*depth_width,(result_ymax-result_ymin)*depth_height));
                                cv::Mat subdepth_full;
                                subdepthP.copyTo(subdepth_full);
                                //cv::rectangle(depth_frame, cv::Point2f(xmin*depth_width, ymin*depth_height), cv::Point2f(xmax*depth_width, ymax*depth_height), 0xffff);
                                cv::Mat subdepth;
                                subdepth_full.convertTo(subdepth,CV_8U,0.0390625);
                                
                                std::vector<std::vector<cv::Point> > contours;
                                std::vector<cv::Vec4i> hierarchy;
                                
                                cv::Scalar m=cv::mean(subdepth);
                                cv::threshold(subdepth,subdepth,m[0]*2.0,100,4);

                                cv::findContours(subdepth,contours,hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
                                for (int i=0; i<contours.size(); i++){
                                    cv::drawContours(subdepth,contours,i,0xffff,cv::FILLED,8,hierarchy, 0, cv::Point());
                                }
                                subdepth.convertTo(subdepth,CV_16U);
                                subdepth=subdepth+subdepth*256;
                                cv::bitwise_and(subdepth,subdepth_full,subdepth_full);
                                cv::Mat mask = cv::Mat(subdepth_full!=0);
                                cv::Scalar avg, dstd;
                                cv::meanStdDev(subdepth_full,avg,dstd,mask);
                                //cv::imshow("ruggero",mask);
                                //cv::waitKey(1);

                                float box_x=avg[0]/1000.0;
                                float box_y=-(((result_xmax+result_xmin)/2.0)*depth_width-cx)/fx*avg[0]/1000.0;
                                float box_z=-(((result_ymax+result_ymin)/2.0)*depth_height-cy)/fy*avg[0]/1000.0;

                                float box_width = avg[0]*depth_width/fx*(result_xmax-result_xmin)/1000.0;
                                float box_height= avg[0]*depth_height/fy*(result_ymax-result_ymin)/1000.0;
                                float box_depth = dstd[0]*2.0/1000.0;

                                //if output as markers is true show cubes in rviz
                                if (output_markers){
                                    marker.header.frame_id = depth_frameid;
                                    marker.header.stamp = ros::Time::now();

                                    marker.ns = "objects_box";
                                    marker.id = kmarker;
                                    marker.type = visualization_msgs::Marker::CUBE;
                                    marker.action = visualization_msgs::Marker::ADD;

                                    marker.pose.position.x = box_x;
                                    marker.pose.position.y = box_y;
                                    marker.pose.position.z = box_z;
                                    marker.pose.orientation.x = 0.0;
                                    marker.pose.orientation.y = 0.0;
                                    marker.pose.orientation.z = 0.0;
                                    marker.pose.orientation.w = 1.0;

                                    marker.scale.x = box_depth;
                                    marker.scale.y = box_width;
                                    marker.scale.z = box_height;

                                    marker.color.r = colorR/255.0;
                                    marker.color.g = colorG/255.0;
                                    marker.color.b = colorB/255.0;
                                    marker.color.a = 0.2f;

                                    marker.lifetime = ros::Duration(markerduration);
                                    markers.markers.push_back(marker);
                                    kmarker++;
                                }

                                //if output as markers is true show flaoting text in rviz
                                if (output_markerslabel){
                                    marker_label.header.frame_id= depth_frameid;
                                    marker_label.header.stamp = ros::Time::now();
                                    marker_label.ns="objects_label";
                                    marker_label.id = kmarker;
                                    marker_label.text = text;
                                    marker_label.type=visualization_msgs::Marker::TEXT_VIEW_FACING;
                                    marker_label.action = visualization_msgs::Marker::ADD;
                                    marker_label.pose.position.x = box_x;
                                    marker_label.pose.position.y = box_y;
                                    marker_label.pose.position.z = box_z+box_height/2.0+0.05;
                                    marker_label.pose.orientation.x=0.0;
                                    marker_label.pose.orientation.y=0.0;
                                    marker_label.pose.orientation.z=0.0;
                                    marker_label.pose.orientation.w=1.0;
                                    marker_label.scale.z=box_height/2;
                                    marker_label.color.r = colorR/255.0;
                                    marker_label.color.g = colorG/255.0;
                                    marker_label.color.b = colorB/255.0;
                                    marker_label.color.a = 0.8f;
                                    marker_label.lifetime = ros::Duration(markerduration);
                                    markers.markers.push_back(marker_label);
                                    kmarker++;
                                }

                                //if you want a topic as list of data
                                if (output_boxlist){
                                    tmp_box.label=text;
                                    tmp_box.confidence=result_confidence;
                                    tmp_box.x=box_x;
                                    tmp_box.y=box_y;
                                    tmp_box.z=box_z;
                                    tmp_box.width=box_width;
                                    tmp_box.height=box_height;
                                    tmp_box.depth=box_depth;
                                    box_list.objectboxes.push_back(tmp_box);
                                }
                            }

                            //if output is a rgb image
                            if (output_as_image){
                                result_xmin*=color_width;
                                result_xmax*=color_width;
                                result_ymin*=color_height;
                                result_ymax*=color_height;

                                //compose a label on the top
                                std::ostringstream c;
                                c << ":" << std::fixed << std::setprecision(3) << result_confidence;
                                cv::putText(frame_now, text + c.str(),cv::Point2f(result_xmin, result_ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,cv::Scalar(colorB, colorG, colorR));
                                cv::rectangle(frame_now, cv::Point2f(result_xmin, result_ymin), cv::Point2f(result_xmax, result_ymax), cv::Scalar(colorB, colorG, colorR));
                            }
                        }
                    }

                }

                //result output
                if (output_as_list){
                    results_list.header.stamp=ros::Time::now();
                    result_pub.publish(results_list);
                    results_list.objects.clear();
                }
                
                //frame output
                if (output_as_image){
                    output_image_msg.header.stamp=ros::Time::now();
                    output_image_msg.header.frame_id=depth_frameid;
                    output_image_msg.height=frame_now.rows;
                    output_image_msg.width=frame_now.cols;
                    output_image_msg.encoding="bgr8";
                    output_image_msg.is_bigendian=false;
                    output_image_msg.step=frame_now.cols*3;
                    size_t size = output_image_msg.step * frame_now.rows;
                    output_image_msg.data.resize(size);
                    memcpy((char*)(&output_image_msg.data[0]), frame_now.data, size);
                    image_pub.publish(output_image_msg);
                }

                //markers output
                if (depth_analysis&&(output_markers||output_markerslabel)){
                    marker_pub.publish(markers);
                }

                //boxlist output
                if (depth_analysis&&output_boxlist){
                    box_list.header.stamp=ros::Time::now();
                    box_list.header.frame_id=depth_frameid;
                    boxlist_pub.publish(box_list);
                }

                //call roscore
                ros::spinOnce();

                //update internal lockers system
                if (is_first_frame) {
                    is_first_frame = false;
                }
                frame_now = frame_next;
                frame_next = cv::Mat();
                engine_curr.swap(engine_next);
                cf_available=false;
                is_last_frame=true;
            }    
        }
    }
    //hey! there is something not working here!
    catch(const std::exception& e){
        ROS_ERROR("%s",e.what());
        return -1;
    }
    return 0;
}
//canova play ramen, gomma tamburo