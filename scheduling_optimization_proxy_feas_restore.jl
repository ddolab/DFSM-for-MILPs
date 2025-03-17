using LinearAlgebra, Random, Gurobi, GAMS, DataFrames, CSV, Printf, BARON, JuMP, Ipopt;
using Distributed
using JuMP, Gurobi
using CPUTime

include("C:/scheduling_optimization_proxy(3).jl") # trains a NN based optimization proxy, change destination to the folder with scheduling_optimization_proxy(2)/(3).jl

global Xtest=zeros((size(arrk,1)),testset,I,V)
global utest=zeros(testset)


utest=u[end-testset+1:end,:]
xt1=zeros(testset,I,V)
xt=zeros(testset,I,V)

for k=1:testset
    for i=1:I
        for j=1:J
            xt[k,i,(j-1)*(T)+1:j*T].=xmain[size(u,1)-testset+k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
        end
            xt[k,i,J*T+1:J*T+J].=xmain[size(u,1)-testset+k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
        for j=1:J
            xt[k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].=xmain[size(u,1)-testset+k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
        end
    end
end

for k=1:testset
    for i=1:I
        xt1[k,i,:].=xt[k,i,:]
        if (i>1)
            xt1[k,i,J*T+J+1:end].=0
        end
    end
end

global cost_test=zeros(testset,I,V)
for k=1:testset
    for i=1:I
        for j=1:J
           global cost_test[k,i,(j-1)*T+1:j*T]=gamijP[size(u,1)-testset+k,i,j,:]
        end
    end
end

global A1=zeros(1,V)
global A2=ones(1,I)
global A3=zeros(J*T-1,V)
global A4=zeros(J,V)
global A5=zeros(I,J,V)
global A6=zeros(1,V)
global A7=zeros(I,1,V)
global A8=zeros(J,T-1,V)
global A9=zeros(J,T,V)


for j=1:J*T
    global A1[1,j]=1
end
for t=1:J*T-1
    if (t%T!=0)
        global A3[t,J*T+J+t]=1
        global A3[t,J*T+J+t+1]=-1
    end
end

for j=1:J
    global A4[j,J*T+J+j*T]=1
end

for j=1:J
    global A5[:,j,J*T+j].=1
end

global A6[1,J*T+1:J*T+J].=1

global A7[:,1,J*T+1:J*T+J].=1

global testerr=zeros(size(arrk,1))
global testobjerr=zeros(size(arrk,1))
global BVR=zeros(size(arrk,1))


### feasible projection for the NN predictions

for l=1:(size(arrk,1))

    for k=1:testset
        for i=1:I
            for j=1:J
                Xtest[l,k,i,(j-1)*(T)+1:j*T].=Xtestval[l,k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
            end
                Xtest[l,k,i,J*T+1:J*T+J].=Xtestval[l,k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
            for j=1:J
                Xtest[l,k,i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].=Xtestval[l,k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
            end
        end
    end

    global Xtest2=zeros(testset,I,V)
    global Xtest3=zeros(testset,I,V)
    global OVt2=zeros(testset)
    
    for k=1:testset
        Q10= Model(optimizer_with_attributes(Gurobi.Optimizer))
        @variable(Q10, 1>=Xijt[1:I,1:J,1:T]>=0, Bin)
        @variable(Q10, Sij[1:I,1:J] >=0)
        @variable(Q10, Tjt[1:J,1:T]>=0)
        @variable(Q10, xpred[1:I,1:V])
        @variable(Q10, pp[1:I,1:V]>=0)
        for i=1:I
            for j=1:J
                @constraint(Q10, xpred[i,(j-1)*(T)+1:j*T].==Xijt[i,j,:])
            end
                @constraint(Q10, xpred[i,J*T+1:J*T+J].==Sij[i,:])
            for j=1:J
                @constraint(Q10, xpred[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J)].==Tjt[j,:])
            end
        end
        for i=1:I
            for j=1:J
                global A5[i,j,(j-1)*T+1:j*T].=-epsi[size(u,1)-testset+k,i]
            end
            for j=1:J
                global A7[i,1,(j-1)*T+1:j*T].=touij[size(u,1)-testset+k,i,j]
            end
            for j=1:J
                for t=1:T-1
                    global A8[j,t,T*(j-1)+t+1]=eta
                    global A8[j,t,T*(j-1)+J*T+J+t]=1
    
                end
                global A8[j,:,J*T+j].=-1
            end
            for j=1:J
                for t=1:T
                    global A9[j,t,T*(j-1)+t]=eta
                    global A9[j,t,T*(j-1)+J*T+J+t]=-1
    
                end
                global A9[j,:,J*T+j].=1
            end
    
        end
        for i=1:I
            @constraint(Q10,A3*xpred[i,:].<=0)
            @constraint(Q10,A4*xpred[i,:].==Tj)
            @constraint(Q10,A1*xpred[i,:].==1)
            @constraint(Q10,A5[i,:,:]*xpred[i,:].<=0)
            @constraint(Q10, A6*xpred[i,:].>=rhoi[size(u,1)-testset+k,i])
            @constraint(Q10, A7[i,:,:]*xpred[i,:].<=epsi[size(u,1)-testset+k,i])
            for j=1:J
                @constraint(Q10, A8[j,:,:]*xpred[i,:].<=eta)
                @constraint(Q10, A9[j,:,:]*xpred[i,:].<=eta-touij[size(u,1)-testset+k,i,j])
            end
        end
        for j=1:J*T
            @constraint(Q10, A2*xpred[:,j].<=1)
        end
        @constraint(Q10, (xpred.-Xtest[l,k,:,:]).<=pp)
        @constraint(Q10, -(xpred.-Xtest[l,k,:,:]).<=pp)
    
    
        
    
        @objective(Q10, Min, sum(pp[:,1:J*T]))
        #@objective(Q10, Min, sum(pp[:,1:J*T]))
        
        optimize!(Q10)
        primal_status(Q10)
        
        if (primal_status(Q10)==NO_SOLUTION)
            
            Xtest2[k,:,:]=zeros(I,V)
            OVt2[k]=0
            Xtest3[k,:]=zeros(I,V)
        else
            
            Xtest2[k,:,:]=value.(xpred)
            OVt2[k]=sum(sum(cost_test[k,i,j].*value.(xpred[i,j]) for j=1:V) for i=1:I)
            Xtest3[k,:,:]=Xtest2[k,:,:]
            for i=2:I
                Xtest3[k,i,J*T+J+1:end].=0
            end
    
        end
    end
    
    

  
    sum2=zeros(testset)
    osum2=zeros(testset)



    for i=1:testset
        sum2[i]=100*(sum(abs.(Xtest3[i,:,:]-xt1[i,:,:])))/(sum(abs.(xt1[i,:,:])))
        osum2[i]=100*(abs.(OVt2[i]-obj[end-testset+i]))/obj[end-testset+i]
    end


    global XX=zeros(testset,I,J)
    global YY=zeros(testset,I,J)
    for k=1:testset
    for i=1:I
        for j=1:J
            XX[k,i,j]=sum(Xtest3[k,i,(j-1)*T+1:j*T])
        end
    end
    end

    for k=1:testset
    for i=1:I
        for j=1:J
            YY[k,i,j]=sum(xt1[k,i,(j-1)*T+1:j*T])
        end
    end
    end


    
    println("Test error : ", sum(sum2)/testset)
    println("Objective Test error : ", sum(osum2)/testset)
    testerr[l]=sum(sum2)/testset
    testobjerr[l]=sum(osum2)/testset
    BVR[l]=sum(abs.(XX.-YY))*100/sum(YY)

end
