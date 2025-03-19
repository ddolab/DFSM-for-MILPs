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


include(".jl")
T=size(z1,2)
Pmax=1
E1=E1/Eo1[1]
Peng1=Peng1/Pmax
z1=z1/swi
xo=hcat(E1,Peng1,z1)
arrk=[10, 20 ,30, 40 ,50, 60, 70, 80, 90,  100, 200, 300, 400, 500]
lc=[:red, :orange, :green, :blue, :violet]
leg=["|K|=10","|K|=20","|K|=30","|K|=40","|K|=50","|K|=60","|K|=70","|K|=80","|K|=90","|K|=100","|K|=200"]#,"|K|=300","|K|=400","|K|=500"]
diffz=zeros(size(arrk,1))
zval=zeros(size(arrk,1),100,T)
Pengval=zeros(size(arrk,1),100,T)
Eval=zeros(size(arrk,1),100,T+1)
p_1_curve=plot()

global nntesterr=zeros(size(arrk,1))
global nntrain=zeros(size(arrk,1))

for j=1:size(arrk,1)
    k=arrk[j]
    ptdes=zeros(T,size(u,1))
    xo1=zeros(3T+1, size(u,1))

    for i=1:size(u,1)
        ptdes[:,i].=Ptdes1[i,:]
        xo1[:,i].=xo[i,:]
    end

    global x_train=zeros(T,k)
    global x_test=zeros(T,100)

    global y_train=zeros(3T+1,k)
    global y_test=zeros(3T+1,100)


    x_train=ptdes[:,1:k]
    y_train=xo1[:,1:k]

    x_test=ptdes[:,end-99:end]
    y_test=xo1[:,end-99:end]



    model=Chain(
        Dense(T, 3T, relu),
        Dense(3T, 6T, relu),
        Dense(6T, 3T+1),
        sigmoid
    )

    loss1(x,y)=mse(model(x),y)
    ps=Flux.params(model)

    learning_rate=0.01
    opt=ADAM(learning_rate)

    loss_history=[]
    epochs=500

    for epoch in 1:epochs
        train!(loss1, ps, [(x_train,y_train)],opt)
        train_loss=loss1(x_train,y_train)
        push!(loss_history, train_loss)
        println("Epoch=$epoch : Training Loss= $train_loss")
    end

    global ytr_hat=model(x_train)
    global y_hat=model(x_test)

    global ytr_hat[2T+2:3T+1,:]=(ytr_hat[2T+2:3T+1,:]*swi)
    global y_hat[2T+2:3T+1,:]=(y_hat[2T+2:3T+1,:]*swi)

    y_train[2T+2:3T+1,:]=y_train[2T+2:3T+1,:]*swi
    y_test[2T+2:3T+1,:]=y_test[2T+2:3T+1,:]*swi
   

    global y=y_test
    
    println("Test accuracy: ",mean(y_hat[2T+2:3T+1,:].==y[2T+2:3T+1,:]))
    println("Training accuracy: ",mean(ytr_hat[2T+2:3T+1,:].==y_train[2T+2:3T+1,:]))
  
    diffz[j]=sum(abs.(y_hat[2T+2:3T+1,:].-y_test[2T+2:3T+1,:]))
    zval[j,:,:]=y_hat[2T+2:3T+1,:]'
    Pengval[j,:,:]=y_hat[T+2:2T+1,:]'*Pmax
    Eval[j,:,:]=y_hat[1:T+1,:]'*Eo1[1]
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
        sum2=zeros(100)
        for kk=1:arrk[j]
            sum1[kk]=100*(sum(abs.(ytr_hat'[kk,:]-y_train'[kk,:])))/(sum(abs.(y_train'[kk,:])))
        end
        for kk=1:100
            sum2[kk]=100*(sum(abs.(y_hat'[kk,:]-y_test'[kk,:])))/(sum(abs.(y_test'[kk,:])))
        end

        global nntrain[j]=sum(sum1)/arrk[j]
        global nntesterr[j]=sum(sum2)/100

end
    
#savefig("C:/plot.png")
