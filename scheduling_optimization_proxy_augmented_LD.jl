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


include("C:/UMN/Research/Summers 2022/Kanpsack Surrogate/main files/Case Study 3/data/file26.jl") ## change destination of input file ##


I=size(Xijt,2) # number of batches
J=size(Xijt,3) # number of units
eta=maximum(Tjt) # Horizon length
T=size(Xijt,4) # number of time slots
V=2J*T+J 


bound=10 #the surrogate parameters are bounded between [-bound,bound]
Tj=eta*ones(J)

global A1n=zeros(I,I*J*T+I*J+J*T)
global A2n=zeros(J*T,I*J*T+I*J+J*T)
global A3n=zeros(J*T-1,I*J*T+I*J+J*T)
global A4n=zeros(I*J,I*J*T+I*J+J*T)
global A5n=zeros(I*J*T,I*J*T+I*J+J*T)
global A6n=zeros(I*J*T,I*J*T+I*J+J*T)
global A7n=zeros(I,I*J*T+I*J+J*T)
global A8n=zeros(I,I*J*T+I*J+J*T)

for i=1:I
    for j=(i-1)*J*T+1:i*J*T
        global A1n[i,j]=1
    end
end

for t=1:J*T 
    for j=1:I
        global A2n[t,t+(j-1)*J*T]=1
    end
end

for t=1:J*T-1
    #if (t%T!=0)
        global A3n[t,I*J*T+I*J+t]=1
        global A3n[t,I*J*T+I*J+t+1]=-1
    #end
end

for i=1:I
    for j=1:J 
        #if (i==j)
            global A4n[(i-1)*J+j,I*J*T+(i-1)*J+j]=1
        #end
    end
end

for i=1:I
    for j=1:J
        #for t=1:T
            global A4n[(i-1)*J+j,(i-1)*J*T+(j-1)*T+1:(i-1)*J*T+(j-1)*T+T].=-100
        #end
    end
end

for i=1:I
    for j=1:J
        global A5n[(i-1)*J+j+1:(i-1)*J+j+T,I*J*T+(i-1)*J+j].=-1
        for t=1:T
            global A5n[(i-1)*J*T+(j-1)*T+t,(i-1)*J*T+(j-1)*T+t]=eta
        end
    end
end

for j=1:J
    for t=2:T
        #global A5[J*T+(j-1)*T+t,(i-1)*J*T+(j-1)*T+t]=1
    end
end

for i=1:I 
    global A7n[i,I*J*T+(i-1)*J+1:I*J*T+(i)*J].=1
end


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
arrk=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#, 200]#, 300, 400, size(u,1)-100]
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

    global y_train1=zeros(I*V,k)
    global y_test1=zeros(I*V,testset)


    #=for k=1:k
        for i=1:I
            for j=1:J
                y_train1[i*(j-1)*(T)+1:i*j*T,k].=xmain1[k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
            end
                y_train1[i*J*T+1:i*J*T+J,k].=xmain1[k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
            for j=1:J
                y_train1[i*(j-1)*(T)+(J*T+J)+1:i*j*T+(J*T+J),k].=xmain1[k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
            end
        end
    end
    
    for k=1:testset
        for i=1:I
            for j=1:J
                y_test1[i,(j-1)*(T)+1:j*T,k].=xmain1[size(u,1)-testset+k,(i-1)*(J*T)+(j-1)*T+1:(i-1)*(J*T)+(j-1)*T+T]
            end
                y_test1[i,J*T+1:J*T+J,k].=xmain1[size(u,1)-testset+k,I*J*T+(i-1)*J+1:I*J*T+(i-1)*J+J]
            for j=1:J
                y_test1[i,(j-1)*(T)+(J*T+J)+1:j*T+(J*T+J),k].=xmain1[size(u,1)-testset+k,I*J*T+I*J+(j-1)*T+1:I*J*T+I*J+(j-1)*T+T]
            end
        end
    end=#

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

    global λ1=zeros(J*T-1)
    global λ2=zeros(J*T)
    global λ3=zeros(I*J)
    global λ4=zeros(I)
    global γ=zeros(I)

    loss(x,y)=mse(model(x),y)+sum(λ1'*sum(max.(0,A3n*model(x)), dims=2)+λ2'*sum(max.(0,A2n*model(x).-1), dims=2)+λ3'*sum(max.(0,A4n*model(x)), dims=2)+λ4'*sum(max.(0,rhoi[1:k,:]'.-A7n*model(x)), dims=2)+γ'*sum(abs.((A1n*model(x).-1)), dims=2))
    #loss1(x)=sum(λ1'*sum(max.(0,A3*model(x)), dims=2)+λ2'*sum(max.(0,A2*model(x).-1), dims=2)+γ'*sum(abs.((A1*model(x).-1)), dims=2))
    ps=Flux.params(model)

    learning_rate=0.0001
    opt=ADAM(learning_rate)

    loss_history=[]
    epochs=1000

    for epoch in 1:epochs
        Flux.train!(loss, ps, [(x_train,y_train)],opt)
        #Flux.train!(loss1, ps, [(x_train)],opt)
        train_loss=loss(x_train,y_train)#+loss1(x_train)

        global λ1=λ1.+0.001*(sum(max.(0,A3n*model(x_train)), dims=2))/(k)  # update at each epoch
        global λ2=λ2.+0.001*(sum(max.(0,A2n*model(x_train).-1), dims=2))/(k)
        global λ3=λ3.+0.001*(sum(max.(0,A4n*model(x_train)), dims=2))/(k)
        #global λ4=λ4.+0.000001*(sum(max.(0,rhoi[1:k,:]'.-A7n*model(x_train)), dims=2))/(k)
        global γ=γ.+0.001*sum(abs.(A1n*model(x_train).-1), dims=2)/(k)
       
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
