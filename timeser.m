
clc;
clear;
close all;
%% input data
input_data=xlsread('3-project_time series data_students.xlsx');
inputSeries = 1:275;
InputSeries=num2cell(inputSeries);
targetSeries = num2cell(input_data(1:275));
%% parameters of the network
feedbackDelays = 1:30;
hiddenLayerSize =[10 10];
trainFcn='trainlm';
%% parameters of the trainlm algorithm
net.trainParam.epochs = 1000;
net.trainParam.mu = 0.001;
net.trainParam.mu_dec = 0.1;
net.trainParam.mu_inc = 10;
%% creating a non linear autoregressive network and preparing the network for training
net = narnet(feedbackDelays,hiddenLayerSize,'none',trainFcn); [inputs,inputStates,layerStates,targets] = preparets(net,{},{},targetSeries'); net.divideParam.trainRatio = 70/100; net.divideParam.valRatio = 15/100; net.divideParam.testRatio = 15/100;
%% Train the Network
rng(9);
[net,tr] = train(net,inputs,targets,inputStates,layerStates);
%% Test the network
[outputs, InpF, TarF] = net(inputs,inputStates,layerStates);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
%% view the network
view(net)
figure, plotperform(tr)
figure, plotresponse(targets,outputs)
figure, ploterrcorr(errors)
%% Closed Loop Network
[netc,xi,ai] = closeloop(net, InpF,TarF);
netc.name = [net.name ' - Closed Loop'];
view(netc)
%% predicting next 30 values of the time series data
[Outputs1,InpF1,TarF1] = netc(cell(0,30),xi,ai); 
test_predict=cell2mat(Outputs1);
%loading next 30 data points for testing prediction results
prompt = 'enter the file name of the test data with .xlsx extension: ';
str = input(prompt,'s');
te_ta=xlsread(str);
%% plotting the predicted data and the target data figure;
plot((1:275),input_data');
hold on
plot((276:305),te_ta');
hold on
plot((276:305),test_predict);
mse=immse(te_ta',test_predict)