clc
clear
close all

DataSize = [ 256, 256, 207 ];
PatchSize = [ 4, 4, 4 ];
PatchStep = [ 1, 1, 1 ];

% % load trained ANN model
caffe.set_mode_cpu();
model_dir = '';
net_model = [model_dir 'deploy.prototxt'];
net_weights = [model_dir 'trained_model_iter_100000.caffemodel'];
phase = 'test';
if ~exist(net_weights, 'file')
  error('Model not found');
end
net = caffe.Net(net_model, net_weights, phase);

% % load normalization parameter
load ybData_2017_normal10_randoms0930_qs_64_matlab settings1 settings2

% % read testing images
MAP1_Path = 'simulatedHRRT_subject20_3_noise_1_conventional_beta_0.001.i';
Fid = fopen( MAP1_Path, 'rb' );
FileData = fread( Fid, inf, 'float' );
MAP1_Data  = reshape( FileData, DataSize );

MAP2_Path = 'simulatedHRRT_subject20_3_noise_1_conventional_beta_0.004.i';
Fid = fopen( MAP2_Path, 'rb' );
FileData = fread( Fid, inf, 'float' );
MAP2_Data  = reshape( FileData, DataSize );

MAP3_Path = 'simulatedHRRT_subject20_3_noise_1_conventional_beta_0.007.i';
Fid = fopen( MAP3_Path, 'rb' );
FileData = fread( Fid, inf, 'float' );
MAP3_Data  = reshape( FileData, DataSize );

fclose( 'all' );

% % generate testing data
% calculate maximum intensity
max_test = max( MAP1_Data(:) );    

Data1 = MAP1_Data ./ max_test;
Data2 = MAP2_Data ./ max_test;
Data3 = MAP3_Data ./ max_test;

PatSize=PatchSize;

DataSize = size( Data1 );
if numel(DataSize)==2
    DataSize(3)=1;
end

RecData  = zeros( DataSize );
CntData  = zeros( DataSize );

ColItrNum = DataSize(2) - PatSize(2) + 1;
FrmItrNum = DataSize(3) - PatSize(3) + 1;

BlkSize = [ DataSize(1)  PatSize(2:3) ];
SlidCnt = countcover( BlkSize, PatSize, [ 1 1 1 ] );

BlkFrm = 1 : PatSize(3);

for FrmIdx = 1 : FrmItrNum
    
    BlkCol = 1 : PatSize(2);
    for ColIdx = 1 : ColItrNum
        
        BlkData1 = Data1( :, BlkCol, BlkFrm );
        BlkData2 = Data2( :, BlkCol, BlkFrm );
        BlkData3 = Data3( :, BlkCol, BlkFrm );
        
        BlkSize = size( BlkData1 );
        
        % extract patches
        VecData1 = PatchExtract( BlkData1, PatSize, [ 1 0 0 ] );
        VecData2 = PatchExtract( BlkData2, PatSize, [ 1 0 0 ] );
        VecData3 = PatchExtract( BlkData3, PatSize, [ 1 0 0 ] );
        
        % remove DC from each patch        
        [ VecData1, DCVal1 ] = remove_dc( VecData1, 'columns' );
        [ VecData2, DCVal2 ] = remove_dc( VecData2, 'columns' );
        [ VecData3, DCVal3 ] = remove_dc( VecData3, 'columns' );
        
        DCVal = DCVal1;
        
        % combine patches from 3 images
        TestData_o = [ VecData1; VecData2; VecData3 ];
        
        % normalize testing input
        TestData_o = mapminmax.apply( TestData_o, settings2 );
        input_data = zeros( 192, 1, 1, 253 );
        input_data( :, 1, 1, :) = TestData_o;
        
        % feed the ANN with input data
        scores = net.forward( {input_data} );
        RecVecData_o = double( scores{1} );
        
        % de-normalize the output of ANN
        RecVecData_o = mapminmax.reverse( RecVecData_o, settings1 );
        
        % add DC back
        RecVecData_o = add_dc( RecVecData_o, DCVal, 'columns' );
        
        RecVecData = RecVecData_o .* max_test;
        
        % arrange patch back to image
        RecComData = PatchCombine( RecVecData, BlkSize, PatSize, [ 1 0 0 ] );
        
        RecData( :, BlkCol, BlkFrm ) = RecData( :, BlkCol, BlkFrm ) + RecComData;
        CntData( :, BlkCol, BlkFrm ) = CntData( :, BlkCol, BlkFrm ) + SlidCnt;
        
        BlkCol = BlkCol + 1;
    end
    
    BlkFrm = BlkFrm + 1;
end

RecData = RecData ./ CntData ;
RecPETData=RecData;

% % save recovered image
RecImg = RecPETData;
PETRecPath = 'recovered_PET_test.i';
Fid = fopen( PETRecPath, 'wb' );
fwrite( Fid, RecImg, 'float' );
fclose( 'all' );
