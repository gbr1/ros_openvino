#This file is part of the ros_openvino package (https://github.com/gbr1/ros_openvino or http://gbr1.github.io).
#Copyright (c) 2019 Giovanni di Dio Bruno.
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.
# 
#You should have received a copy of the GNU Affero General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.

#This message is for one 2D object detection result

#which object is detected
string label

#confidence of result
float32 confidence

#normalized value of box top-left point coordinate x
float32 x

#normalized value of box top-left point coordinate y
float32 y

#normalized value of box height
float32 height

#normalized value of box width
float32 width
