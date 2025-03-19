using LinearAlgebra, Random, Gurobi, GAMS, DataFrames, CSV, Printf, BARON, JuMP, Ipopt;
using Distributed
using JuMP, Gurobi, CPUTime
using Graphs
using Flux, Images, MLDatasets, Plots, Statistics
using Flux: crossentropy, mse, onecold, onehotbatch, train!
using LinearAlgebra, Random, Statistics
using Base.Iterators: repeated, partition
using LaTeXStrings
#import MathProgIncidence

include("C:/UMN/Research/Summers 2022/Kanpsack Surrogate/main files/e2e/T=30/t30data_file26.jl")

T=30
Emax=100
Pmax=1
S=swi

arrk=[10, 20 ,30, 40 ,50, 60, 70, 80, 90,  100, 200]#, 300, 400, 500]
lc=[:red, :orange, :green, :blue, :violet]
leg=["|K|=10","|K|=20","|K|=30","|K|=40","|K|=50","|K|=60","|K|=70","|K|=80","|K|=90","|K|=100","|K|=200"]#,"|K|=300","|K|=400","|K|=500"]
diffz=zeros(size(arrk,1))
zval=zeros(size(arrk,1),100,T)
Pengval=zeros(size(arrk,1),100,T)
Eval=zeros(size(arrk,1),100,T+1)

p_1_curve=plot()
#Eo1=Eo1/Emax
#Ptdes1=Ptdes1/Emax
#Peng1=Peng1/Emax
#Pmax=Pmax/Emax
#Pbat1=Pbat1
#E1=E1/Emax
#z1=z1/S
#Emax=Emax/100


nV=3T+1
nC=2T+1
fV=zeros(size(z1,1),nV,4)
fC=zeros(size(z1,1),nC,2)
A=zeros(size(z1,1),nC,nV)
X=zeros(size(z1,1),nV)

for i=1:size(z1,1)
    X[i,1:T+1].=E1[i,:]
    X[i,T+2:2T+1].=Peng1[i,:]
    X[i,2T+2:3T+1].=z1[i,:]
end

for i=1:size(z1,1)
    fV[i,:,1].=normalize(cc)
    fV[i,:,2].=0
    fV[i,1:T+1,3].=Emax
    fV[i,T+2:2T+1,3].=Pmax
    fV[i,2T+2:3T+1,3].=S

    fV[i,1:2T+1,4].=0
    fV[i,2T+2:3T+1,4].=1
end


fC[:,1,1].=Eo1[1]
fC[:,1,2].=0

fC[:,2:T+1,1].=0
fC[:,2:T+1,2].=1

fC[:,T+2:2T+1,2].=1

for i=1:size(z1,1)
    fC[i,T+2:2T+1,1].=-Ptdes1[i,:]*100
end

tou=5


A[:,1,1].=1
for i=1:T 
    A[:,i+1,i].=-100/tou
    A[:,i+1,i+1].=100/tou     ## -Et/tou + Et+1/tou -Pteng <= -Ptdes
    A[:,i+1,T+1+i].=-100  
end

for i=1:T
    A[:,T+1+i,1*(T)+1+i].=1   # Peng<=z Pmax
    A[:,T+1+i,2*(T)+1+i].=-Pmax/swi
end

global nntesterr=zeros(size(arrk,1))
global nntrain=zeros(size(arrk,1))


for j=1:size(arrk,1)

    K=arrk[j]

    Xtr1=zeros(K,nV,4)
    Xtr2=zeros(K,nC,2)
    Xtr3=zeros(K,nC,nV)

    ytr=zeros(K,nV)


    Xtr1[:,:,:].=fV[1:K,:,:]
    Xtr1=permutedims(Xtr1,[3,2,1])
    Xtr2[:,:,:].=fC[1:K,:,:]
    Xtr2=permutedims(Xtr2,[3,2,1])
    Xtr3[:,:,:].=A[1:K,:,:]
    Xtr3=permutedims(Xtr3,[3,2,1])

    ytr.=X[1:K,:]
    ytr=permutedims(ytr,[2,1])


    Xts1=zeros(100,nV,4)
    Xts2=zeros(100,nC,2)
    Xts3=zeros(100,nC,nV)

    yts=zeros(100,nV)

    Xts1[:,:,:].=fV[end-99:end,:,:]
    Xts1=permutedims(Xts1,[3,2,1])
    Xts2[:,:,:].=fC[end-99:end,:,:]
    Xts2=permutedims(Xts2,[3,2,1])
    Xts3[:,:,:].=A[end-99:end,:,:]
    Xts3=permutedims(Xts3,[3,2,1])

    yts.=X[end-99:end,:]
    yts=permutedims(yts,[2,1])


    embs=8

    Cmodel = Chain(
        Flux.Dense(size(Xtr2,1), embs, relu),
        Flux.Dense(embs, embs),
    )

    Vmodel= Chain(
        Flux.Dense(size(Xtr1,1), embs, relu),
        Flux.Dense(embs, embs),
    )

    ConvC= Chain(
        Flux.Dense(embs*2,embs,relu),
        Flux.Dense(embs, embs),
    )

    global cFeat=zeros(embs,nC,K)
    global vFeat=zeros(embs,nV,K)


        for i=1:K 
            global cFeat[:,:,i]=Vmodel(Xtr1)[:,:,i]*Xtr3[:,:,i]
            global vFeat[:,:,i]=Cmodel(Xtr2)[:,:,i]*permutedims(Xtr3,[2,1,3])[:,:,i]
        end

        vFeat=ConvC(vcat(vFeat,Vmodel(Xtr1)))
        cFeat=ConvC(vcat(cFeat,Cmodel(Xtr2)))

    Xtrain=zeros(size(vcat(reshape(vFeat,:,size(vFeat,3)),reshape(cFeat,:,size(cFeat,3))),1),K)
    Xtrain=vcat(reshape(vFeat,:,size(vFeat,3)),reshape(cFeat,:,size(cFeat,3)))

    model1=Chain(
        Flux.Dense(size(vcat(reshape(vFeat,:,size(vFeat,3)),reshape(cFeat,:,size(cFeat,3))),1),91),
    )




    #Xtr=vcat(Cmodel(Xtr2), Vmodel(Xtr1), Emodel(Xtr3))
    #Xts=vcat(Cmodel(Xts2), Vmodel(Xts1), Emodel(Xts3))


    loss(x,y)=mse(model1(x),y)
        ps=Flux.params(model1,ConvC,Cmodel,Vmodel)

        learning_rate=0.001
        opt=ADAM(learning_rate)

        loss_history=[]
        epochs=500

        for epoch in 1:epochs
            train!(loss, ps, [(Xtrain,ytr)],opt)
            train_loss=loss(Xtrain,ytr)
            push!(loss_history, train_loss)
            println("Epoch=$epoch : Training Loss= $train_loss")
        end


        global ytr_hat=model1(Xtrain)

        global cFeats=zeros(embs,nC,100)
        global vFeats=zeros(embs,nV,100)


        for i=1:100 
            global cFeats[:,:,i]=Vmodel(Xts1)[:,:,i]*Xts3[:,:,i]
            global vFeats[:,:,i]=Cmodel(Xts2)[:,:,i]*permutedims(Xts3,[2,1,3])[:,:,i]
        end

        vFeats=ConvC(vcat(vFeats,Vmodel(Xts1)))
        cFeats=ConvC(vcat(cFeats,Cmodel(Xts2)))

        Xtest=zeros(size(vcat(reshape(vFeats,:,size(vFeats,3)),reshape(cFeats,:,size(cFeats,3))),1),K)
        Xtest=vcat(reshape(vFeats,:,size(vFeats,3)),reshape(cFeats,:,size(cFeats,3)))


        global y_hat=model1(Xtest)

        diffz[j]=sum(abs.(round.(y_hat[2T+2:3T+1,:]).-yts[2T+2:3T+1,:]))
        zval[j,:,:]=round.(y_hat[2T+2:3T+1,:]')
        Pengval[j,:,:]=y_hat[T+2:2T+1,:]'
        Eval[j,:,:]=y_hat[1:T+1,:]'

        gr(size=(600,600))
        font = Plots.font("Helvetica", 20)
        plot!(1:epochs, loss_history,
            xlabel= "epochs",
            ylabel= "training loss",
            titel= "Learning Curve",
            #label=leg[j],
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
                sum1[kk]=100*(sum(abs.(round.(abs.(ytr_hat'[kk,2T+2:3T+1]))-ytr'[kk,2T+2:3T+1])))/(sum(abs.(ytr'[kk,2T+2:3T+1])))
            end
            for kk=1:100
                sum2[kk]=100*(sum(abs.(round.(abs.(y_hat'[kk,2T+2:3T+1]))-yts'[kk,2T+2:3T+1])))/(sum(abs.(yts'[kk,2T+2:3T+1])))
            end

            global nntrain[j]=sum(sum1)/arrk[j]
            global nntesterr[j]=sum(sum2)/100
        
end