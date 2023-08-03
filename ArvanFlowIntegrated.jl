
using Plots
using Statistics
using LinearAlgebra
using Printf
using CUDA
using JLD2
using ImageFiltering

use_GPU = true #switch between GPU and CPU, global
todevice(x) = use_GPU ? CuArray(x) : x #convert arrays to device

#function to return the distance between two points
function sq_distance(A=Float32[0,0],B=Float32[0,0])
    return ((B[1] - A[1])^2 + (B[2] - A[2])^2)
end

#defines the solid and fluid nodes of an arbitrary cylinder
function cylinder(SB, FB, bndry_out, bndry_in, X_lat_d, Y_lat_d, centerX, centerY, radius) 
    start_X = Int(floor((centerX - radius)*X_lat_d))
    end_X = Int(ceil((centerX + radius)*X_lat_d))
    start_Y = Int(floor((centerY - radius)*Y_lat_d))
    end_Y = Int(ceil((centerY + radius)*Y_lat_d))

    for j = start_X:end_X
        for k = start_Y:end_Y
            if sq_distance([j/X_lat_d,k/Y_lat_d],[centerX,centerY]) < radius^2
                    SB[k,j] = 1        
                    FB[k,j] = 0  
            end
        end
    end    
    #The code below finds the inner and outer boundary of the cylinder for force processing. Inaugurated by arjav garg & hriday G.
    for j = (start_X - 3):(end_X + 3)
        for k = (start_Y-3):(end_Y + 3)
            if SB[k,j] == 1
                n = FB[k-1:k+1, j-1:j+1]
                if 1 in n
                    bndry_in[k,j] = 1
                end
            end
            if FB[k,j] == 1
                n = SB[k-1:k+1, j-1:j+1]
                if 1 in n
                    bndry_out[k,j] = 1
                end
            end
            
        end
    end
    return SB, FB, bndry_out, bndry_in
end

#defines the solid and fluid nodes of an arbitrary airfoil
function symmetric_NACA_airfoil(SB, FB, bndry_out, bndry_in, X_lat_d, Y_lat_d, startX, startY, thickness, chord) 
    start_X = Int(floor(start_X*X_lat_d))
    end_X = Int(ceil((chord+start_X)*X_lat_d))
    start_Y = Int(floor((centerY - radius)*Y_lat_d))
    end_Y = Int(ceil((centerY + radius)*Y_lat_d))

    for j = start_X:end_X
        for k = start_Y:end_Y
            if sq_distance([j/X_lat_d,k/Y_lat_d],[centerX,centerY]) < radius^2
                    SB[k,j] = 1        
                    FB[k,j] = 0  
            end
        end
    end    
    #The code below finds the inner and outer boundary of the cylinder for force processing. Inaugurated by arjav garg & hriday G.
    for j = (start_X - 3):(end_X + 3)
        for k = (start_Y-3):(end_Y + 3)
            if SB[k,j] == 1
                n = FB[k-1:k+1, j-1:j+1]
                if 1 in n
                    bndry_in[k,j] = 1
                end
            end
            if FB[k,j] == 1
                n = SB[k-1:k+1, j-1:j+1]
                if 1 in n
                    bndry_out[k,j] = 1
                end
            end
            
        end
    end
    return SB, FB, bndry_out, bndry_in
end

#defines the solid and fluid nodes of an arbitrary square
function square(SB, FB, bndry_out, bndry_in, X_lat_d, Y_lat_d, centerX, centerY, length) 
    hlen = length/2 #half length of side
    start_X = Int(floor((centerX - hlen)*X_lat_d))
    end_X = Int(ceil((centerX + hlen)*X_lat_d))
    start_Y = Int(floor((centerY - hlen)*Y_lat_d))
    end_Y = Int(ceil((centerY + hlen)*Y_lat_d))
    SB[start_Y:end_Y,start_X:end_X].= 1
    FB[start_Y:end_Y,start_X:end_X].= 0
    
        #The code below finds the inner and outer boundary of the cylinder for force processing. Inaugurated by arjav garg & hriday G.
    for j = (start_X - 3):(end_X + 3)
        for k = (start_Y-3):(end_Y + 3)
            if SB[k,j] == 1
                n = FB[k-1:k+1, j-1:j+1]
                if 1 in n
                    bndry_in[k,j] = 1
                end
            end
            if FB[k,j] == 1
                n = SB[k-1:k+1, j-1:j+1]
                if 1 in n
                    bndry_out[k,j] = 1
                end
            end        
        end
    end
    return SB, FB, bndry_out, bndry_in
end

#defines the solid and fluid nodes of a wall on domain edges
function wall_BC(SB, FB, X_lat_d, Y_lat_d, orientation, side) 
    #orientation false is horizontal, true is vertical
    #side false is left/up, side true is right/down

    if orientation && side
        SB[:,end] .= 1
        FB[:,end] .= 0
    elseif orientation && !side
        SB[:,1] .= 1
        FB[:,1] .= 0
    elseif !orientation && side
        SB[1,:] .= 1
        FB[1,:] .= 0
    elseif !orientation && !side
        SB[end,:] .= 1
        FB[end,:] .= 0
    end
    return SB, FB
end

#defines the particle distribution of an absorbing wall on domain edges
function absorb_boundary(F, orientation, side) 
    #orientation false is horizontal, true is vertical
    #side false is left/up, side true is right/down

    if orientation && side
        F[:,end,[7,8,9]] .= F[:,end-1,[5,4,3]]
    elseif orientation && !side
        F[:,1,[5,4,3]] .= F[:,2,[7,8,9]]
    elseif !orientation && side
        F[1,:,[9,2,3]] .= F[2,:,[7,6,5]]
    elseif !orientation && !side
        F[end,:,[7,6,5]] .= F[end-1,:,[9,2,3]]
    end
    return F
end

#defines the particle distribution of a velocity boundary on domain edges (zou-he)
function velocity_boundary(F, orientation, side, ux, uy, rho, weights, cx, cy, c)
    #orientation false is horizontal, true is vertical
    #side false is left/up, side true is right/down

    if orientation && side #right wall
            rho[:,end] = (2*(F[:,end,3] .+ F[:,end,4] .+ F[:,end,5]) .+ F[:,end,1] .+ F[:,end,2] .+ F[:,end,6]) ./ (1 + ux)
            NeqF = weights[4]*( (3*(c^-2)*(cx[4]-cx[8])*ux) + 4.5*(((cx[4]*ux)^2 - (cx[8]*ux)^2)*(c^(-4))) ) * rho[:,end]  #non-equilibrium part
            F[:,1,9] = 0.5* ( - ux*rho[:,end] + uy*rho[:,end] + 2*F[:,end,5] + NeqF + F[:,end,6] - F[:,end,2])
            F[:,1,7] = 0.5* (- ux*rho[:,end] - uy*rho[:,end] + 2*F[:,end,3] + NeqF - F[:,end,6] + F[:,end,2] )
            F[:,1,8] = F[:,1,4] - NeqF 


    elseif orientation && !side #left wall
            rho[:,1] = (2*(F[:,1,7] .+ F[:,1,8] .+ F[:,1,9]) .+ F[:,1,1] .+ F[:,1,2] .+ F[:,1,6]) ./ (1 - ux)
            NeqF = weights[4]*( (3*(c^-2)*(cx[4]-cx[8])*ux) + 4.5*(((cx[4]*ux)^2 - (cx[8]*ux)^2)*(c^(-4))) ) * rho[:,1]  #non-equilibrium part
            F[:,1,3] = 0.5* (ux*rho[:,1] + uy*rho[:,1] + 2*F[:,1,7] - NeqF + F[:,1,6] - F[:,1,2])
            F[:,1,5] = 0.5* (ux*rho[:,1] - uy*rho[:,1] + 2*F[:,1,9] - NeqF - F[:,1,6] + F[:,1,2] )
            F[:,1,4] = F[:,1,8] + NeqF 

    elseif !orientation && side #bottom wall
            rho[1,:] = (2*(F[1,:,7] .+ F[1,:,6] .+ F[1,:,5]) .+ F[1,:,1] .+ F[1,:,4] .+ F[1,:,8]) ./ (1 - uy)
            NeqF = weights[2]*( (3*(c^-2)*(cy[2]-cy[6])*uy) + 4.5*(((cy[2]*uy)^2 - (cy[6]*uy)^2)*(c^(-4))) ) * rho[1,:]  #non-equilibrium part
            F[1,:,3] = 0.5* (ux*rho[1,:] + uy*rho[1,:] + 2*F[1,:,7] - NeqF + F[1,:,8] - F[1,:,4])
            F[1,:,9] = 0.5* (ux*rho[1,:] - uy*rho[1,:] + 2*F[1,:,5] - NeqF - F[1,:,8] + F[1,:,4] )
            F[1,:,2] = F[1,:,6] + NeqF 

    elseif !orientation && !side #top wall 
            rho[end,:] = (2*(F[end,:,2] .+ F[end,:,3] .+ F[end,:,9]) .+ F[end,:,1] .+ F[end,:,4] .+ F[end,:,8]) ./ (1 + uy)
            NeqF = weights[2]*( (3*(c^-2)*(cy[2]-cy[6])*uy) + 4.5*(((cy[2]*uy)^2 - (cy[6]*uy)^2)*(c^(-4))) ) * rho[end,:]  #non-equilibrium part
            F[end,:,5] = 0.5* (ux*rho[end,:] - uy*rho[end,:] + 2*F[end,:,9] + NeqF + F[end,:,8] - F[end,:,4])
            F[end,:,7] = 0.5* (- ux*rho[end,:] - uy*rho[end,:] + 2*F[end,:,3] + NeqF - F[end,:,8] + F[end,:,4] )
            F[end,:,6] = F[1,:,2] - NeqF 
    end
    return F, rho
end

#defines the particle distribution of a pressure boundary on domain edges (zou-he)
function pressure_boundary(F, orientation, side, ux, uy, rho0, weights, cx, cy, c, wall_flag)  
    #orientation false is horizontal, true is vertical
    #side false is left/up, side true is right/down

    if orientation && side #right wall
            ux[:,end] = ((2*(F[:,end,3] .+ F[:,end,4] .+ F[:,end,5]) .+ F[:,end,1] .+ F[:,end,2] .+ F[:,end,6]) ./ (rho0)) .- 1
            NeqF = weights[4]* rho0 *( (3*(c^-2)*(cx[8]-cx[4]).*ux[:,end]) + 4.5*(((cx[8].*ux[:,end]).*(cx[8]*ux[:,end]) - (cx[4].*ux[:,end]).*(cx[4].*ux[:,end]))*(c^(-4))))   #non-equilibrium part
            F[:,end,9] = 0.5* ( - rho0*ux[:,end] +  2*F[:,end,5] - NeqF + F[:,end,6] - F[:,end,2])
            F[:,end,7] = 0.5* (- rho0*ux[:,end]  + 2*F[:,end,3] - NeqF - F[:,end,6] + F[:,end,2] )
            F[:,end,8] = F[:,end,4] + NeqF 


    elseif orientation && !side #left wall
            ux[:,1] = 1 .- (2*(F[:,1,7] .+ F[:,1,8] .+ F[:,1,9]) .+ F[:,1,1] .+ F[:,1,2] .+ F[:,1,6]) ./ (rho0)
            NeqF = weights[4]* rho0 *( (3*(c^-2)*(cx[8]-cx[4]).*ux[:,1]) + 4.5*(((cx[8].*ux[:,1]).*(cx[8]*ux[:,1]) - (cx[4].*ux[:,1]).*(cx[4].*ux[:,1]))*(c^(-4))) )   #non-equilibrium part
            F[:,1,3] = 0.5* (rho0*ux[:,1] + 2*F[:,1,7] + NeqF + F[:,1,6] - F[:,1,2])
            F[:,1,5] = 0.5* (rho0*ux[:,1] + 2*F[:,1,9] + NeqF - F[:,1,6] + F[:,1,2] )
            F[:,1,4] = F[:,1,8] - NeqF 

    elseif !orientation && side #bottom wall
            uy[1,:] = 1 .- (2*(F[1,:,7] .+ F[1,:,6] .+ F[1,:,5]) .+ F[1,:,1] .+ F[1,:,4] .+ F[1,:,8]) ./ (rho0)
            NeqF = weights[2]* rho0 .*( (3*(c^-2)*(cy[2]-cy[6])*uy[1,:]) + 4.5*(((cy[2]*uy[1,:]).*(cy[2]*uy[1,:]) - (cy[6].*uy[1,:]).*(cy[6]*uy[1,:]))*(c^(-4))) )   #non-equilibrium part
            F[1,:,3] = 0.5* ( rho0*uy[1,:] + 2*F[1,:,7] - NeqF + F[1,:,8] - F[1,:,4])
            F[1,:,9] = 0.5* ( rho0*uy[1,:] + 2*F[1,:,5] - NeqF - F[1,:,8] + F[1,:,4] )
            F[1,:,2] = F[1,:,6] + NeqF 

    elseif !orientation && !side #top wall 
            uy[end,:] = ((2*(F[end,:,2] .+ F[end,:,3] .+ F[end,:,9]) .+ F[end,:,1] .+ F[end,:,4] .+ F[end,:,8]) ./ rho0) .- 1
            NeqF = weights[2]* rho0 *( (3*(c^-2)*(cy[2]-cy[6])*uy[1,:]) + 4.5*(((cy[2]*uy[1,:]).*(cy[2]*uy[1,:]) - (cy[6]*uy[1,:]).*(cy[6]*uy[1,:]))*(c^(-4))) )  #non-equilibrium part
            F[end,:,5] = 0.5* ( - rho0*uy[end,:] + 2*F[end,:,9] + NeqF + F[end,:,8] - F[end,:,4])
            F[end,:,7] = 0.5* ( - rho0*uy[end,:] + 2*F[end,:,3] + NeqF - F[end,:,8] + F[end,:,4] )
            F[end,:,6] = F[1,:,2] - NeqF 
    end
    return F, ux, uy
end

#computes collisions, returns particle distribution function
function collision(ux, uy, F, Feq, bndryF, tau, c, cxs, cys, rho, weights, FB ,Q, scheme_type, force_type, MomentMatrix, MomentMatrixInv, Force_x, Force_y, c_t, S, use_LES, smag_c, filter_width)
    
    if cmp(scheme_type, "SRT") == 0 && !use_LES
        for n = 1:Q
            Feq[:,:,n] = rho[:,:].*weights[n].*(-1.5*(ux.*ux + uy.*uy)*(c^(-2)) + 4.5*(cxs[n]*ux + cys[n]*uy).*(cxs[n]*ux + cys[n]*uy)*(c^(-4)) + 3*(cxs[n]*ux + cys[n]*uy)*(c^-2) .+ 1)
        end
        F = (F .- (1.0/tau)*(F .- Feq) ).*FB + bndryF
        F[F .< 0] .= 0

    elseif cmp(scheme_type, "MRT") == 0 && !use_LES

        m = zeros(size(F))
        meq = zeros(size(F))

        rho_velocity_squared_sum = rho.*(ux.*ux + uy.*uy)
        X_momentum = rho .* ux
        Y_momentum = rho .* uy

        Ny = Int32(size(F,1))
        Nx = Int32(size(F,2))

        meq = cat(rho , -2*(rho) .+ 3*rho_velocity_squared_sum, - 3*rho_velocity_squared_sum .+ rho,  X_momentum , - X_momentum, Y_momentum, -Y_momentum, rho.*(ux.*ux - uy.*uy), rho .* (ux .* uy), dims = 3)

        W = MomentMatrixInv*S
        
        m = reshape(transpose(MomentMatrix * transpose(reshape(F, (Nx*Ny,Q)))), (Ny,Nx,Q))          #multiply moment matrix with F
        F = F .- reshape(transpose(W * transpose(reshape((m .- meq), (Nx*Ny,Q)))), (Ny,Nx,Q))       #retrieve F

        F = F .* FB + bndryF
        F[F .< 0] .= 0
    end

    if cmp(scheme_type, "SRT") == 0 && use_LES
        for n = 1:Q
            Feq[:,:,n] = rho[:,:].*weights[n].*(-1.5*(ux.*ux + uy.*uy)*(c^(-2)) + 4.5*(cxs[n]*ux + cys[n]*uy).*(cxs[n]*ux + cys[n]*uy)*(c^(-4)) + 3*(cxs[n]*ux + cys[n]*uy)*(c^-2) .+ 1)
        end

        F_neq = F .- Feq
        SQ = todevice(zeros(size(F_neq[:,:,1])))                          #Strain rate primitive in explicit formula to calculate eddy viscosity, zhang et al 2018 doi:10.1016/j.camwa.2018.01.019; Li, Wang 2010 https://doi.org/10.1155/2010/724578
                                                                         
        for n = 1:Q
                SQ = cys[n]*cxs[n]*F_neq[:,:,n]
        end

        rho0 = mean(rho)                                                                               
        nu_eddy = todevice(18*((smag_c.*filter_width).^2).*sqrt.(2*sum(SQ.*SQ)./(rho0*(c^4))))  #calculate eddy viscosity
        tau_eddy = todevice(0.5*(sqrt.(tau .* tau + nu_eddy) .- tau))
        tau = tau .+ tau_eddy

        F = (F .- (1.0 ./tau).*(F .- Feq) ).*FB + bndryF
        F[F .< 0] .= 0
    
    end

    Sx = Array(zeros(size(F))) #source term in x direction
    Sy = Array(zeros(size(F))) #source term in y direction

    if cmp(force_type, "None") == 0

        i = 1 #dummy assignment for no forces

    elseif cmp(force_type, "Guo") == 0                  #Guo et al 2002
        
        for i = 1:Q
            Sx[:,:,i] = ((1 - c_t/(2*tau))*weights[i]*((-ux[:,:] .+ cxs[i]) .+ (cxs[i]*cxs[i].*ux[:,:]))).*Force_x[:,:]
            Sy[:,:,i] = ((1 - c_t/(2*tau))*weights[i]*((-uy[:,:] .+ cys[i]) .+ (cys[i]*cys[i].*uy[:,:]))).*Force_y[:,:]
            F[:,:,i] = F[:,:,i] .+  Sx[:,:,i] .+ Sy[:,:,i]
        end

    elseif cmp(force_type, "ShanChen") == 0             #Shan and Chen, 1993; dummy because no force
        
        k = 0

    end
    
    return F
end

#initializes flow field
function initialize_flow_field(ux, uy, F, bndryF, FB, c, cxs, cys, rho,  weights, Q, type)

    if cmp(type, "SRT_simple") == 0
        for n = 1:Q
            F[:,:,n] = rho[:,:].*weights[n].*(-1.5*(ux.*ux + uy.*uy)*(c^(-2)) + 4.5*(cxs[n]*ux + cys[n]*uy).*(cxs[n]*ux + cys[n]*uy)*(c^(-4)) + 3*(cxs[n]*ux + cys[n]*uy)*(c^-2) .+ 1)
        end
        F = F.*FB + bndryF
        F[F .< 0] .= 0
    end
    
    return F
end
#calculates forces on a body when outer and inner boundaries are already known, using the corrected momentum exchange method (Caiazzo et al, 2009)
function calculate_forces(F, bndry_in, bndry_out, cxs, cys, Q)
    F_in = F .* bndry_in
    F_out = F .* bndry_out 

    #boundary links, flow from fluid to solid is denoted as positive, boundarys are (y,x)
    #inout = bndry_out - bndry_in


    F_in_sum = zeros(Q)
    F_out_sum = zeros(Q)
    lift = 0
    drag = 0
    for n = 1:Q
        F_in_sum[n] = sum(F_in[:,:,n])
        F_out_sum[n] = sum(F_out[:,:,n])

        lift = lift + (F_out_sum[n] - F_in_sum[n]) * cys[n] 
        drag = drag + (F_out_sum[n] - F_in_sum[n]) * cxs[n] 


    end
    
    return lift, drag
end
 #solid boundary - pressure integration for corner nodes

#set the wall flag
function set_wall_flag(SB)
    wall_flag = [0,0,0,0] #flag to note walls
    if SB[1,1] == 1 && SB[1,end] == 1 #bottom
        wall_flag = wall_flag + [1,0,0,0]
    end
    if SB[1,end] == 1 && SB[end,end] == 1 #right
        wall_flag = wall_flag + [0,1,0,0]
    end
    if SB[end,1] == 1 && SB[end,end] == 1 #top
        wall_flag = wall_flag + [0,0,1,0]
    end
    if SB[end,1] == 1 && SB[1,1] == 1 #left
        wall_flag = wall_flag + [0,0,0,1]
    end
    return wall_flag
end

#set the force density according to the density function of the domain
function set_force_density(Force_x, Force_y, global_force_x, global_force_y, local_force_x, local_force_y, rho)

    Force_x = (local_force_x .+ global_force_x) ./ rho
    Force_y = (local_force_y .+ global_force_y) ./ rho
    return Force_x, Force_y
end

#plots contours
function contours(var1, var2, FB, Ny, Nx, min_vort, max_vort, min_vel, max_vel, lift, drag) 
    var1[FB .< 0.5,1] .= NaN
    var2[FB .< 0.5,1] .= NaN
    hm1 = heatmap(var1, ylimits = (0,Ny), xlimits = (0,Nx),  clim = (min_vort,max_vort), aspect_ratio = :equal, title = "vorticity contours", c= :thermometer )
    hm2 = heatmap(var2, ylimits = (0,Ny), xlimits = (0,Nx),  clim = (min_vel,max_vel), aspect_ratio = :equal, title = "velocity contours", c= :ice)
    p1 = plot( [lift], title = "Lift coefficient", ylimits = (-0.5,0.5), legend = false)
    p2 = plot( [drag], title = "Drag coefficient",  ylimits = (-0.2,2.5), legend = false)
    plot(hm1, hm2, p1, p2, layout = (4,1))
    #:thermometer :ice  
    gui()
end

#defines smagorinsky constants for each cell in the domain
function set_smagorinsky(SB, FB, smagorinsky_BL, smagorinsky_FS, smooth_type, smooth_iter, smooth_radius)

    if cmp(smooth_type, "nosmooth_FS") == 0 || cmp(smooth_type, "nosmooth_BL") == 0
        print("Setting smagorinsky constant field with type \"", smooth_type, "\".\n")
    else
        print("Setting smagorinsky constant field with type \"", smooth_type, "\" applied ", smooth_iter," times.\n")
    end
    
    #smooth_type sets the type of smoothing: gaussian, nosmooth_FS, nosmooth_BL
    #smooth_iter sets the number of times the 'blur' is applied
    #smooth_radius sets the width of the blur: [3, 5, 7]
    
    if cmp(smooth_type, "nosmooth_FS") == 0 #applies free-stream smagorinsky constant to all pixels (very high Re flows)
        smag_c = Array(smagorinsky_FS*(SB + FB))

    elseif cmp(smooth_type, "nosmooth_BL") == 0 #applies boundary_layer smagorinsky constant to all pixels (small width channel flows)
        smag_c = Array(smagorinsky_BL*(SB + FB))

    elseif cmp(smooth_type,"gaussian") == 0     #sets up the matrix for smoothing
        smag_c = Array(smagorinsky_BL*SB + smagorinsky_FS*FB)

    else 
        print("No such smoothing function exists! applying nosmooth_FS.\n")
        smag_c = Array(smagorinsky_FS*(SB + FB))

    end

    for i = 1:smooth_iter                       #applies smoothing for the number of iterations specified

        if cmp(smooth_type,"gaussian") == 0 #isotropic gaussian convolution on all cells
            smag_c = imfilter(smag_c, Kernel.gaussian(smooth_radius))

        end
    end
    return smag_c
end

#this function calculates turbulence
function calculate_turbulence(ux_mean, uy_mean, ux, uy, FB)


    
    u_x_fluc = ux .- ux_mean                                                        #fluctuations in X velocity
    u_y_fluc = uy .- uy_mean                                                        #fluctuations in Y velocity
    u_fluc = sqrt.(0.5*(u_x_fluc.^2 + u_y_fluc.^2))                                 #fluctuations total
    
    TKE = (u_x_fluc.^2 + u_y_fluc.^2)                                               #turbulent kinetic energy
    U = sqrt.(ux.^2 + uy.^2)                                                        #velocity magnitude
    T_intensity  =  u_fluc  ./ (U .+ 1E-10)                                         #turbulent intensity
    T_intensity  = T_intensity .* todevice(FB)

    return TKE, T_intensity

end


function ArvanFlow(X_len_real, Y_len_real, X_lat_d, SB, FB, saved_filename, t, Nt, tau, Re)

    #solver settings
    GUI = false #true is on, false is off
    save = true #true is on, false is off. Saves velocity, vorticity & rho. Not implemented yet
    use_GPU = true #switch between GPU and CPU
    use_LES = true #true is on, false is off
    error = false #catch errors


    #saved_filename = "cylinder_test"
    save_frequency = 10000

    #domain settings

    #X_len_real = 5.2 #X length in meters
    #Y_len_real = 0.8 #Y length in meters

    #X_lat_d = 100 # X lattice density per m
    Y_lat_d = X_lat_d # Y lattice density per m
    dx = 1/X_lat_d
    dy = dx
    c_t = 0 #time conversion factor
    dt = 1
    #t = 200 #time in seconds
    param_method = "tau-Re" #specifies parameter setting method, "tau-Re" or "physical"
    X_len = 0 #X length in lattice units
    Y_len = 0 #Y length in lattice units
    smagorinsky_BL = 0.1 #boundary layer smagorinsky constant
    smagorinsky_FS = 0.2 #free stream smagorinsky constant
    filter_width_base = dx #base filter width


    
    c = 1 #lattice speed


    nu_real = 0 #kinematic viscosity in real units
    nu = 0 #kinematic viscosity in lattice units
    #tau = 0
    rho0 = 0
    #Nt = 50000 #number of timesteps
    DTI = 0 #initial domain turbulent intensity
    Vx = 0
    Vy = 0
    simulation_type = "SRT"
    force_type = "None"
    g_real = 0*rho0 #force density due to gravity in real units
    g = g_real*dx/(c_t*c_t) #lattice unit conversion

    if cmp(param_method, "physical") == 0
        nu_real = 0.0003 
        nu = nu_real *dt/(dx*dy)

        tau = nu*3 + 0.5 #tau
        if tau > 1
            print("Warning, tau is greater than unity, (",Floaat16(tau),")  solver may be inaccurate.\n")
        elseif tau < 0.53
            print("Warning, tau approaching 0.5, (",Floaat16(tau),")  solver may be inaccurate.\n")
        end
        rho0 = 100 #average density
        Vx = 1 #velocity
        Vy = 0
        cylinder_rad = 5 #cylinder radius or square length
        cylinder_rad_real = 1*dx
        Re = Float16(sqrt(Vx^2 + Vy^2)*cylinder_rad_real*2/nu_real)
        print("Reynolds number: ", Int16(floor(Re)), "\n")

    elseif cmp(param_method, "tau-Re") == 0
        rho0 = 1000
        nu = (tau - 0.5)/3
        
        nu_real = 1E-3 #viscosity of water 
        c_h = dx                                                    #length scale conversion factor
        c_t = nu*(c_h*c_h)/nu_real                                  #time scale conversion factor
        c_u = c_h/c_t                                               #velocity scale conversion factor
        c_rho = rho0                                                #density scale conversion factor
        c_f = c_rho*c_h/(c_t*c_t)                                   #force scale conversion factor

        c_h_frmt = @sprintf "%.2E" c_h                              #length scale conversion factor
        c_t_frmt = @sprintf "%.2E" c_t                              #time scale conversion factor
        c_u_frmt = @sprintf "%.2E" c_u                              #velocity scale conversion factor
        c_rho_frmt = @sprintf "%.2E" c_rho                          #density scale conversion factor
        c_f_frmt = @sprintf "%.2E" c_f                              #force scale conversion factor

        print("Lattice length scale conversion factor: ",  c_h_frmt, "\n")
        print("Lattice time scale conversion factor: ", c_t_frmt, "\n")
        print("Lattice velocity scale conversion factor: ", c_u_frmt, "\n")
        print("Lattice density conversion factor:", c_rho_frmt, "\n")
        print("Lattice force conversion factor:", c_f_frmt, "\n")
        
        cylinder_rad_real = 0.3
        cylinder_rad = cylinder_rad_real/c_h
        Vx = Re*nu_real/(cylinder_rad_real*2)
        LVx = Vx/c_u #lattice unit conversions
        Re_L = (cylinder_rad*2*LVx)/(nu)
        kolmogorov_length_scale = cylinder_rad_real/Re
        print("Kolmogorov length scale is: ", kolmogorov_length_scale, "\n")

        if kolmogorov_length_scale < c_h
            print("Turbulence is not fully resolved by mesh. Turbulence modeling may be required.\n")
            percentage_diff_kls = @sprintf "%.0f" (abs(kolmogorov_length_scale - c_h)/c_h)*100
            print("Kolmogorov length scale is ", percentage_diff_kls, "% smaller than the grid scale.\n")
        end
        X_len = X_len_real/c_h
        Y_len = Y_len_real/c_h
        print("Reynolds number: ", Int16(floor(Re)), "\n")
        print("Velocity: ", Vx , "\n")
        if abs((Re_L - Re)/Re) > 0.1
            print("Warning! Reynolds numbers in lattice units and real units don't match!\n")
            print("Real Reynolds numeber: ", Re, "\n")
            print("Lattice reynolds number: ", Re_L, "\n")
            readline()
        end
    end

    #discrete velocities, weights, and mesh

    Q = 9
    cxs     = Float32[0, 0, c, c, c, 0, -c, -c, -c] #x discrete velocities
    cys     = Float32[0, c, c, 0, -c, -c, -c, 0, c] #y discrete velocities
    weights = Float32[4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36] #lattice weights
    Nx = Int32(floor(X_len_real * X_lat_d)) #Number of X lattice points
    Ny = Int32(floor(Y_len_real * Y_lat_d)) #Number of Y lattice points
    S = Matrix{Float64}(undef, 9, 9)


    #MRT/SRT definitions

    if use_LES && cmp(simulation_type, "SRT") == 0                                          #Section of code to model LES for SRT
        print("Using LES!\n")
        tau = todevice(tau*ones(Ny,Nx))
    end

    if use_LES && cmp(simulation_type, "MRT") == 0                                          #Section of code to model LES for MRT
        print("Using LES!\n")
        tau = tau*ones(Ny,Nx)
        w_v = 1 ./tau
        w_q = 1.2
        w_e = 1.1
        w_eps = 1

        S = zeros(Ny, Nx, 81)
        S = reshape(S, (Ny, Nx, 9, 9))

        for i = 1:Ny
            for j = 1:Nx
                S[i,j,:,:] = Diagonal([0, w_e, w_eps, 0, w_q, 0, w_q, w_v[i,j], w_v[i,j]])
            end
        end

        S = todevice(S)
        #for i = 1:Ny
            #for j = 1:Nx
    end

    if !use_LES && cmp(simulation_type,"MRT") == 0                                         #define matrix for MRT without LES
        w_v = 1/tau                                                                        #shear viscosity relaxation rate
        w_q = 1.2                                                                          #bulk viscosity relaxation rate (assumed shear viscosity = bulk viscosity)
        w_e = 1.1
        w_eps = 1

        S = Diagonal([0, w_e, w_eps, 0, w_q, 0, w_q, w_v, w_v])
        S = todevice(S)
    end
    
    #Define moment matrix and moment matrix inverse
    MomentMatrix = Matrix{Float64}(undef, 9, Q)

    for i = 1:Q
        MomentMatrix[1,i] = 1
        MomentMatrix[2,i] = -4 + 3*(cxs[i]*cxs[i] + cys[i]*cys[i])
        MomentMatrix[3,i] = 4 - 10.5*(cxs[i]*cxs[i] + cys[i]*cys[i])  + 4.5*(cxs[i]*cxs[i] + cys[i]*cys[i])*(cxs[i]*cxs[i] + cys[i]*cys[i])
        MomentMatrix[4,i] = cxs[i]
        MomentMatrix[5,i] = (-5 + 3*(cxs[i]*cxs[i] + cys[i]*cys[i]))*cxs[i]
        MomentMatrix[6,i] = cys[i]
        MomentMatrix[7,i] = (-5 + 3*(cxs[i]*cxs[i] + cys[i]*cys[i]))*cys[i]
        MomentMatrix[8,i] = cxs[i]*cxs[i] - cys[i]*cys[i]
        MomentMatrix[9,i] = cxs[i]*cys[i]
    end

    MomentMatrixInv = inv(MomentMatrix[:,:])
    MomentMatrix = todevice(MomentMatrix)
    MomentMatrixInv = todevice(MomentMatrixInv)




    #initial conditions

    F = todevice(ones(Ny, Nx, Q))  #Calculate initial velocity field


    #SB = zeros(Ny,Nx) #Initialize solid boundaries; zero = liquid, one = solid
    #FB = ones(Ny,Nx) #Initialize fluid domain; zero = solid, one = liquid
    cy_bndry_out = zeros(Ny,Nx)
    cy_bndry_in = zeros(Ny,Nx)


    #fluid loop
    if use_GPU
        print("Using GPU!\nCreating GPU arrays.\n")
    end


        
    bndryF = todevice(zeros(size(F)))                                     #boundary particle distribution
    rho    = todevice(zeros(Ny,Nx))                                       #density matrix in lattice units
    ux     = todevice(zeros(Ny,Nx))                                       #X velocity in lattice units
    uy     = todevice(zeros(Ny,Nx))                                       #Y velocity in lattice units
    Force_x = todevice(zeros(Ny, Nx))                                     #X force density in lattice units
    Force_y = todevice(zeros(Ny, Nx))                                     #Y force density in lattice units
    Feq = todevice(zeros(size(F)))                                        #equilibrium particle distribution
    local_force_x = todevice(zeros(Ny,Nx))                                #local force in the x direction, useful for fluid structure interaction   
    local_force_y = todevice(zeros(Ny,Nx))                                #local force in the y direction, useful for fluid structure interaction
    lift_time_history = zeros(1)                                          #matrix for lift time histories
    drag_time_history= zeros(1)                                           #matrix for drag time historiess
    TKE_time_history = zeros(1)                                           #matrix for TKE time histories
    TI_time_history = zeros(1)                                            #matrix for TI time histories
    cxs_device = todevice(cxs)                                            #convert cxs to device
    cys_device = todevice(cys)                                            #convert cys to device

    iter = zeros(1)
    if use_GPU
        print("GPU arrays created.\n")
    end

    #initialize flow field
    print("Initializing case...\n")
    print("Setting initial density...\n")

    rho = dropdims(sum(F, dims=3),dims=3) #rho = rho + F[:,:,indx]

    print("Initial density set.\n")
    print("Creating mesh...\n")
    #SB, FB, cy_bndry_out, cy_bndry_in = cylinder(SB, FB, cy_bndry_out, cy_bndry_in, X_lat_d, Y_lat_d, 1.2, 0.4, cylinder_rad_real)
    wall_flag = set_wall_flag(SB)
    
    #LES definitions
    smag_c = todevice(set_smagorinsky(SB, FB, smagorinsky_BL, smagorinsky_FS, "gaussian", 3, 10))
    filter_width = todevice(filter_width_base*ones(Ny, Nx))


    SB_device = todevice(SB) #convert SB to GPU array if needed after scalar indexing
    FB_device = todevice(FB) #convert FB to GPU array if needed after scalar indexing
    cy_bndry_in_device = todevice(cy_bndry_in) #convert inside boundary to GPU array if needed after scalar indexing
    cy_bndry_out_device = todevice(cy_bndry_out) #convert outside boundary to GPU array if needed after scalar indexing
    print("Mesh created.\n")

    print("Initializing flow field...\n")
    bndryF = F .* SB_device
    bndryF = bndryF[:,:, [1, 6, 7, 8, 9, 2, 3, 4, 5]]
    LVx = Vx/c_u #lattice unit conversions
    LVy = Vy/c_u
    if LVx > 0.2
        print("Lattice velocity too large! Simulation may be unstable.\nLattice velocity: ", LVx, " \n")
    elseif LVx < 0.05
        print("Lattice velocity too small! Simulation may be unstable.\nLattice velocity: ", LVx, " \n")
    end
    
    F, ux, uy = pressure_boundary(F, true, true, ux, uy, 0.99*rho0, weights, cxs, cys, c, wall_flag)

    F, rho = velocity_boundary(F, true, false, LVx, LVy, rho, weights, cxs, cys, c)
    F = initialize_flow_field(ux, uy, F, bndryF, FB_device, c, cxs, cys, rho,  weights, Q, "SRT_simple")

    F = F*c_rho./rho

    #=
    for indx = 1:Q
        F[:,:,indx]  = F[:,:,indx]*rho0./rho
    end
    =#
    print("Flow field initialized.\n")
    print("Initialization complete.\n")

    t_array = [1.0,0.0,0.0,0.0,0.0]
    lift_array = [1.0,0.0,0.0,0.0]
    drag_array = [1.0,0.0,0.0,0.0]
    ux_array = [ux, ux, ux, ux, ux]
    uy_array = [uy, uy, uy, uy, uy]
    ux_mean = todevice(ux)
    uy_mean = todevice(uy)
    TKE = todevice(zeros(size(ux)))
    TI = todevice(zeros(size(ux)))
    Lift = 0
    Drag = 0
    Lift_coefficient = 0
    Drag_coefficient = 0
    TKE_av = 0
    TI_av = 0
    raw_lift = 0
    raw_drag = 0
    F_neqerr = 0
    print("Starting solver...\n")
    for it = 1:Nt

        t = mean(t_array[t_array .> 0])[1]
            #TUI display and error handling
        F_neqerr_frmt = @sprintf "%.2E" F_neqerr #convert it to scientific notation
        Lift_frmt = @sprintf "%.3f" Lift_coefficient
        Drag_frmt = @sprintf "%.3f" Drag_coefficient
        TKE_frmt = @sprintf "%.2E" TKE_av
        TI_frmt = @sprintf "%.2E" TI_av


        iter = append!(iter, it)
        days = Int32(floor(t*(Nt-it)/86400))
        hours = Int32(floor(t*(Nt-it)/3600) - 24*days)
        minutes = Int32(floor(t*(Nt-it)/60) - 60*hours - 24*60*days) #calculate time in days, hours & minutes

        if minutes == 0 #make minutes 1 just before simulaation finishes
            minutes = 1
        end
        if it%10 == 1
            print("\nIteration \tTime \t \tNon equilibrium F \tTKE \t \tTI \n \n") #show headings every 10 iterations
        end
        if isnan(F_neqerr)
            F_neqerr_frmt = "Crashed!" #show "crashed!" instead of NaN
        end
        print(Int32(it), "\t \t", days, ":", hours, ":", minutes, "\t \t", F_neqerr_frmt, "\t \t", TKE_frmt, "\t \t", TI_frmt, "\n") # display values
        if isnan(F_neqerr)
            print("\nSimulation ended with exit code 1. Press any key to exit.") #catch error for crashed code
            error = true
            break()
        end
        ttemp = CUDA.@elapsed begin
        
            
                #show GUI loop
                if GUI
                    vorticity = Array((circshift(ux,(-1,0)) - circshift(ux,(1,0))) - (circshift(uy,(0,-1)) - circshift(uy,(0,1))))
                    velocity = Array((ux .* ux + uy .* uy) * dx^2 ./(c_t^2))
                    contours( vorticity, velocity, FB_device, Ny, Nx, -0.1, 0.1, -0.1*(Vx*Vx), 1.2*(Vx*Vx), lift_time_history, drag_time_history)
                end

                for n = 1:Q
                    #streaming step
                    F[:,:,n] = circshift(F[:,:,n], (0,cxs[n])) 
                    F[:,:,n] = circshift(F[:,:,n], (cys[n],0)) 

                end    
                #F, rho = velocity_boundary(F, true, false, 0.15, 0, rho, weights, cxs, cys, c)

                F, ux, uy = pressure_boundary(F, true, true, ux, uy, 0.99*rho0, weights, cxs, cys, c, wall_flag)

                F, rho = velocity_boundary(F, true, false, LVx, LVy, rho, weights, cxs, cys, c)


                bndryF = F .* SB_device
                bndryF = bndryF[:,:, [1, 6, 7, 8, 9, 2, 3, 4, 5]]

                rho = dropdims(sum(F, dims=3),dims=3) #rho = rho + F[:,:,indx]
                
                ux = reshape(transpose(transpose(cxs_device) * transpose(reshape(F, (Nx*Ny,Q)))), (Ny,Nx)) #ux = ux + (F[:,:,n] * cxs[n])
                uy = reshape(transpose(transpose(cys_device) * transpose(reshape(F, (Nx*Ny,Q)))), (Ny,Nx)) #uy = uy + (F[:,:,n] * cys[n]) 

                ux = ux ./rho 
                uy = uy ./rho 

                if cmp(force_type,"None") != 0 
                    Force_x, Force_y = set_force_density(Force_x, Force_y, g, 0, local_force_x, local_force_y, rho) #set force matrix
                end

                #add forces to velocity if the forcetype is not "None"
                if cmp(force_type, "Guo") == 0                      #Guo et al 2002
                    ux = ux .+ (Force_x*c_t)./(2 * rho)
                    uy = uy .+ (Force_y*c_t)./(2 * rho) 

                elseif cmp(force_type, "ShanChen") == 0             #Shan and Chen, 1993
                    ux = ux .+ tau*c_t*(Force_x*c_t)./( rho)
                    uy = uy .+ tau*c_t*(Force_y*c_t)./( rho) 

                end

            

                ux = ux .* FB_device
                uy = uy .* FB_device
                #collision

                F_prec = mean(F)
                F = collision(ux, uy, F, Feq, bndryF, tau, c, cxs, cys, rho, weights, FB_device, Q, simulation_type, force_type, MomentMatrix, MomentMatrixInv, Force_x, Force_y, c_t, S, use_LES, smag_c, filter_width) #0.03 seconds
                #=
                if cmp(simulation_type,"MRT") == 0
                    F = todevice(F) #convert back to GPU array
                end
                =#
                F_mean = mean(F)
                F_neqerr = abs(F_mean - F_prec)/F_mean         #Fneqerr is to calculate the stability of the simulation


                #find lift and drag and convert to physical units
                raw_lift, raw_drag = calculate_forces(F, cy_bndry_in_device, cy_bndry_out_device, cxs, cys, Q)
                Lift = Float32(raw_lift*c_f*c_h*c_h/(c_rho))
                Drag = Float32(raw_drag*c_f*c_h*c_h/(c_rho))
                #4 timestep moving average
                lift_array = circshift(lift_array, 1)
                lift_array[1] = Lift 
                drag_array = circshift(drag_array, 1)
                drag_array[1] = Drag
                Lift = mean(lift_array)[1]
                Lift_coefficient = Lift/(0.5*cylinder_rad_real*2*rho0*(Vx^2+Vy^2))
                Drag = mean(drag_array)[1]
                Drag_coefficient = Drag/(0.5*cylinder_rad_real*2*rho0*(Vx^2+Vy^2))
                lift_time_history = append!(lift_time_history, Lift_coefficient)
                drag_time_history= append!(drag_time_history, Drag_coefficient)

                ux_array = circshift(ux_array, 1)
                ux_array[1] = ux
                uy_array = circshift(uy_array, 1)
                uy_array[1] = uy
                ux_mean = todevice(mean(ux_array))
                uy_mean = todevice(mean(uy_array))

                TKE, TI = calculate_turbulence(ux_mean, uy_mean, ux, uy, FB)

                TKE_av = mean(TKE)
                TI_av = mean(TI)

                TKE_time_history = append!(TKE_time_history, TKE_av)
                TI_time_history = append!(TI_time_history, TI_av)

                if save && (it%save_frequency == 0)
                    print("Saving...\n")
                    num_digits_max = ndigits(Nt)
                    num_digits_iters = ndigits(it)
                    format_it = string(it)
                    format_it = lpad(format_it, num_digits_max - num_digits_iters, '0')
                    save_object(saved_filename*"-F-"*format_it*".jld2" , Array(F))
                    save_object(saved_filename*"-TKE_time_history-"*format_it*".jld2" , Array(TKE_time_history))
                    save_object(saved_filename*"-TI_time_history-"*format_it*".jld2" , Array(TI_time_history))
                    print("Saved!\n")
                end

            end #this is end to measure time
        t_array = circshift(t_array, 1)
        t_array[1] = ttemp
        
    end
    
    if !error
        print("Simulation complete!\n")
        if save
            print("Writing final files...\n")
            save_object(saved_filename*"-F-endwrite.jld2" , Array(F))
            save_object(saved_filename*"-TKE_time_history-endwrite.jld2" , Array(TKE_time_history))
            save_object(saved_filename*"-TI_time_history-endwrite.jld2" , Array(TI_time_history))
            print("Saved!\n")
        end
    else
        print("Encountered an error. Saving TKE average as 1000 and TI average as 100")
        TKE_av = 1000
        TI_av = 100
    end
    return TKE_av, TI_av, error
end