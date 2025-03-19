using Flux, Images, MLDatasets, Plots
using Flux: crossentropy, mse, onecold, onehotbatch, train!
using LinearAlgebra, Random, Statistics
using LaTeXStrings
plot_font = "Computer Modern"

default(
    fontfamily=plot_font,
    linewidth=2, 
)


Random.seed!(1)


#include("C:/data_file.jl") ## change destination of input file ##


I=size(Xijt,2) # number of batches
J=size(Xijt,3) # number of units
eta=maximum(Tjt) # Horizon length
T=size(Xijt,4) # number of time slots
V=2J*T+J 


bound=10 #the surrogate parameters are bounded between [-bound,bound]
Tj=eta*ones(J)


global xmain=zeros(size(u,1),I*J*T+I*J+J*T)
global xmain1=zeros(size(u,1),I*J*T+I*J+J*T)
global Inp1=zeros(size(u,1),I*J+I+I)

for k=1:size(u,1)
    for i=1:I
        for j=1:J
            for t=1:T
                xmain[k,(i-1)*(J*T)+(j-1)*T+t]=Xijt[k,i,j,t]
                xmain1[k,(i-1)*(J*T)+(j-1)*T+t]=Xijt[k,i,j,t]
            end
            xmain[k,I*J*T+(i-1)*J+j]=Sij[k,i,j]
            xmain1[k,I*J*T+(i-1)*J+j]=Sij[k,i,j]/maximum(Sij)
            for t=1:T
                xmain[k,I*J*T+I*J+(j-1)*T+t]=Tjt[k,j,t]
                xmain1[k,I*J*T+I*J+(j-1)*T+t]=Tjt[k,j,t]/eta
            end
        end
    end

    for i=1:I
        for j=1:J
            Inp1[k,(i-1)*(J)+j]=touij[k,i,j]
        end
        Inp1[k,I*(J)+i]=rhoi[k,i]
        Inp1[k,I*(J)+I+i]=epsi[k,i]
    end

end

xo=xmain1
arrk=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, size(u,1)-100]
lc=[:red, :orange, :green, :blue, :violet]
leg=["|K|=10","|K|=20","|K|=30","|K|=40","|K|=50","|K|=60","|K|=70","|K|=80","|K|=90","|K|=100","|K|=200"]#,"|K|=300","|K|=400","|K|=500"]

p_1_curve=plot()

global nntesterr=zeros(size(arrk,1))
global nntrain=zeros(size(arrk,1))

testset=50

global Xtestval=zeros(size(arrk,1),testset,I*J*T+I*J+J*T)

### training a NN based optimization proxy for the given batch of data set

for j=1:size(arrk,1)
    k=arrk[j]
    inp=zeros(I*J+I+I,size(u,1))
    xo1=zeros(I*J*T+I*J+J*T, size(u,1))

    for i=1:size(u,1)
        inp[:,i].=Inp1[i,:]
        xo1[:,i].=xo[i,:]
    end

    global x_train=zeros(I*J+I+I,k)
    global x_test=zeros(I*J+I+I,testset)

    global y_train=zeros(I*J*T+I*J+J*T,k)
    global y_test=zeros(I*J*T+I*J+J*T,testset)


    x_train=inp[:,1:k]
    y_train=xo1[:,1:k]

    x_test=inp[:,end-testset+1:end]
    y_test=xo1[:,end-testset+1:end]



    model=Chain(
        Dense(I*J+I+I, 3*(I*J+I+I), relu),
        Dense(3*(I*J+I+I), 6*(I*J+I+I), relu),
        Dense(6*(I*J+I+I), I*J*T+I*J+J*T),
        sigmoid
    )

    loss(x,y)=mse(model(x),y)
    ps=Flux.params(model)

    learning_rate=0.0001
    opt=ADAM(learning_rate)

    loss_history=[]
    epochs=500

    for epoch in 1:epochs
        train!(loss, ps, [(x_train,y_train)],opt)
        train_loss=loss(x_train,y_train)
        push!(loss_history, train_loss)
        println("Epoch=$epoch : Training Loss= $train_loss")
    end

    global ytr_hat=model(x_train)
    global y_hat=model(x_test)

    global ytr_hat[I*J*T+1:I*J*T+I*J,:]=(ytr_hat[I*J*T+1:I*J*T+I*J,:]*maximum(Sij))
    global ytr_hat[I*J*T+I*J+1:I*J*T+I*J+J*T,:]=(ytr_hat[I*J*T+I*J+1:I*J*T+I*J+J*T,:]*eta)

    global y_hat[I*J*T+1:I*J*T+I*J,:]=(y_hat[I*J*T+1:I*J*T+I*J,:]*maximum(Sij))
    global y_hat[I*J*T+I*J+1:I*J*T+I*J+J*T,:]=(y_hat[I*J*T+I*J+1:I*J*T+I*J+J*T,:]*eta)

    y_train[I*J*T+1:I*J*T+I*J,:]=y_train[I*J*T+1:I*J*T+I*J,:]*maximum(Sij)
    y_train[I*J*T+I*J+1:I*J*T+I*J+J*T,:]=y_train[I*J*T+I*J+1:I*J*T+I*J+J*T,:]*eta

    y_test[I*J*T+1:I*J*T+I*J,:]=y_test[I*J*T+1:I*J*T+I*J,:]*maximum(Sij)
    y_test[I*J*T+I*J+1:I*J*T+I*J+J*T,:]=y_test[I*J*T+I*J+1:I*J*T+I*J+J*T,:]*eta

    global y=y_test
    
  
    gr(size=(600,600))
    font = Plots.font("Helvetica", 20)
    plot!(1:epochs, loss_history,
        xlabel= "epochs",
        ylabel= "training loss",
        titel= "Learning Curve",
        label=leg[j],
        linewidth=2,
        alpha=0.4,
        titlefontsize=24,
        guidefontsize=24,
        tickfontsize=20,
        legendfontsize=18,
        )
        display(p_1_curve)

        sum1=zeros(arrk[j])
        sum2=zeros(testset)
        for kk=1:arrk[j]
            sum1[kk]=100*(sum(abs.(ytr_hat'[kk,:]-y_train'[kk,:])))/(sum(abs.(y_train'[kk,:])))
        end
        for kk=1:testset
            sum2[kk]=100*(sum(abs.(y_hat'[kk,:]-y_test'[kk,:])))/(sum(abs.(y_test'[kk,:])))
        end

        global nntrain[j]=sum(sum1)/arrk[j]
        global nntesterr[j]=sum(sum2)/testset
        global Xtestval[j,:,:]=y_hat'
end
    
#savefig("plot.png")
