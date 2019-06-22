load('palmar.mat')
load('lateral.mat')

F0= 60 ; Fs = 1000 ; Q=35;
W0= 2*F0 /Fs;
Bw = W0 / Q;
n = 4; %degree of notch
[a,b] = iircomb(n,Bw);
palmar = filter(a,b,palmar);
lateral = filter(a,b,lateral);

C1_train = palmar([1:100],:);
C2_train = lateral([1:100],:);
C1_test = palmar([101:150],:);
C1_test_s = cat(2,C1_test,(2*ones(50,1))); %add symbol 2 to C1 at column 3001
 C2_test = lateral([101:150],:);
 C2_test_s = cat(2,C2_test,(3*ones(50,1))); %add symbol 3 to C2
 test = cat(1,C1_test_s,C2_test_s);
 size(test)
 
 %to make shuffle function
 row_no = randperm(100,100); %number of rows random indexing
 test_shuffled = test((row_no(1,1)),:); %first shuffled row
 for n =2:1 :100
test_shuffled = cat(1,test_shuffled,test((row_no(1,n)),:));
 end
size(test_shuffled)


%F1 represent energy, M represent mean, S represent standard deviation
%F1_C1 given class
F1_C1 = sum(C1_train.^2,2);
MF1_C1 = mean(F1_C1); %mean = sum of column cells/n
SF1_C1 = std(F1_C1); %std =root( sum(x-mean)^2/(n-1))
%F2_C1
F2_C1 = sum(C1_train.^4,2);
MF2_C1 = mean(F2_C1);
SF2_C1 = std(F2_C1);
%F3_C1
row_length= size(C1_train , 1);
coloumn_length= size(C1_train , 2) ;
for row = 1:row_length
for col = 3:coloumn_length
equation= ( - C1_train(row , col) * C1_train(row , col-2) + C1_train(row , col-1)^2) ;
Palmar_results(col-2) = equation ;
end
Nonlinear_Energy_Palmar(row)= sum(Palmar_results);
end
F3_C1 = Nonlinear_Energy_Palmar';
MF3_C1 = mean(F3_C1);
SF3_C1 = std(F3_C1);
%F4_C1
row_length = size(C2_train , 1);
coloumn_length = size(C2_train , 2) ;
for row = 1:row_length
for col = 2:coloumn_length
equation = (  C2_train(row , col) - C2_train(row , col-1)) ;
Palmar_results(col-1) = equation ;
end
curve_length_Palmar(row) = sum(Palmar_results);
end
F4_C1 = curve_length_Palmar';
MF4_C1 = mean(F4_C1);
SF4_C1 = std(F4_C1);

%F1_C2
F1_C2 = sum(C2_train.^2,2);
MF1_C2 = mean(F1_C2);
SF1_C2 = std(F1_C2);
%F2_C2
F2_C2 = sum(C2_train.^4,2);
MF2_C2 = mean(F2_C2);
SF2_C2 = std(F2_C2);
%F3_C2
row_length= size(C2_train , 1);
coloumn_length= size(C2_train , 2) ;
for row = 1:row_length
for col = 3:coloumn_length
equation= ( - C2_train(row , col) * C2_train(row , col-2) + C2_train(row , col-1)^2) ;
lateral_results(col-2) = equation ;
end
Nonlinear_Energy_lateral(row)= sum(lateral_results);
end
F3_C2 = Nonlinear_Energy_lateral';
MF3_C2 = mean(F3_C2);
SF3_C2 = std(F3_C2);
%F4_C2
row_length = size(C2_train , 1);
coloumn_length = size(C2_train , 2) ;
for row = 1:row_length
for col = 2:coloumn_length
equation = ( C2_train(row , col) - C2_train(row , col-1)) ;
lateral_results(col-1) = equation ;
end
curve_length_lateral(row) = sum(lateral_results);
end
F4_C2= curve_length_lateral';
MF4_C2 = mean(F4_C2);
SF4_C2 = std(F4_C2);

%testing part
n_true = 0; %representing at the end of next for loop how many right 
%decisions from the 100 decisions

for m = 1:1:100
x = test_shuffled(m,[1:3000]);%row to be tested
c_n = test_shuffled(m,3001);%number ofsymbol indicating class whether 2 or 3
F1 = sum(x.^2,2);
F2 = sum(x.^4,2);
%F3
for col = 3:1:3000
equation= ( - test_shuffled(m , col) * test_shuffled(m , col-2) + test_shuffled(m , col-1)^2);
lateral_results(col-2) = equation ;
end
F3= sum(lateral_results);
%F4
for col = 2:1:3000
equation = (  test_shuffled(m , col) - test_shuffled(m , col-1)) ;
Palmar_results(col-1) = equation ;
end
F4 = sum(Palmar_results);

F1_C1_T = normpdf(F1,MF1_C1,SF1_C1);
F2_C1_T = normpdf(F2,MF2_C1,SF2_C1);
F3_C1_T = normpdf(F3,MF3_C1,SF3_C1);
F4_C1_T = normpdf(F4,MF4_C1,SF4_C1);
C1_F = [0.5] * F1_C1_T * F2_C1_T*F3_C1_T*F4_C1_T;


F1_C2_T = normpdf(F1,MF1_C2,SF1_C2);
F2_C2_T = normpdf(F2,MF2_C2,SF2_C2);
F3_C2_T = normpdf(F3,MF3_C2,SF3_C2);
F4_C2_T = normpdf(F4,MF4_C2,SF4_C2);
C2_F = [0.5] * F1_C2_T * F2_C2_T*F3_C2_T*F4_C2_T;


if (C1_F> C2_F)
CS_expected = 2;%CS Class representing symbol found
else
CS_expected = 3;
end

%compare the found class from algorithm to the real one
if(CS_expected==c_n)
    n_true= n_true +1;
end

end

n_true
accuracy = n_true/100

