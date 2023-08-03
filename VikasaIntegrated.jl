#this is the genetic algorithm program which helps generate meshes for ArvanFlow

using Plots
using Statistics
using StatsBase
using LinearAlgebra
using Printf
using CUDA
using JLD2
include("ArvanFlowIntegrated.jl")

#generate a random floating point number
function randfloat(a, b)
    return rand() * (b - a) + a
end

#generate a bezier curve
function bezier_curve(control_points, t)
    n = length(control_points) - 1
    result = [0.0, 0.0]
    for i in 0:n
        binomial_coefficient = binomial(n, i)
        bernstein_polynomial = binomial_coefficient * (1 - t)^(n - i) * t^i
        result += bernstein_polynomial * control_points[i + 1]
    end
    return result
end

#calculates fitness based on objectives and weights
function calculate_fitness(objectives, weights)
    fitness = dot(objectives, weights)
    return fitness
end

#creates the genes for the population at the start of the simulation
function generate_population(X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size)

    X_int_start = Int32(floor(X_start_point*20))
    X_int_end = Int32(floor(X_end_point*20))

    Y_int_start = Int32(floor(Y_start_point*20))
    Y_int_end = Int32(floor(Y_end_point*20))


    for p = 1:pop_size

        #generate X control points
        CP_X_1 = rand( min(X_int_start, X_int_end):max(X_int_start, X_int_end) )/20 
        CP_X_2 = rand( min(X_int_end, X_int_start):max(X_int_end, X_int_start) )/20 

        #generate Y control points
        CP_Y_1 = rand( min(Y_int_start, Y_int_end):max(Y_int_start, Y_int_end) )/20 
        CP_Y_2 = rand( min(Y_int_end, Y_int_start):max(Y_int_end, Y_int_start) )/20 

        #make genes
        genes[p,:] = [CP_X_1, CP_Y_1, CP_X_2, CP_Y_2]

    end
    return genes
end

#mutates the genes of the population
function mutate(mutation_rate, mutation_value, X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size, mutation_type)

    #calculate minimum and maximum values of X
    min_X = min(X_start_point, X_end_point)
    max_X = max(X_start_point, X_end_point)

    #calculate minimum and maximu values of Y
    min_Y = min(Y_start_point, Y_end_point)
    max_Y = max(Y_start_point, Y_end_point)

    #iterate through all genes
    if cmp(mutation_type, "addsub") == 0
        mut_array = rand(pop_size, 4)

        mut_array[mut_array .<= mutation_rate] .= mutation_value
        mut_array[mut_array .> mutation_rate] .= 0

        for g = 1:4
            for p = 1:pop_size
                mut_array[p,g] = mut_array[p,g]*rand([-1,1])
            end
        end
        

        genes = genes .+ mut_array
        
        #make sure that the genes are within the domain size
        for p = 1:pop_size
            for g_x in [1,3]
                if genes[p, g_x] < min_X

                    genes[p, g_x] = min_X

                elseif genes[p, g_x] > max_X

                    genes[p, g_x] = max_X
                end
            end

            for g_y in [2,4]
                if genes[p, g_y] < min_Y

                    genes[p, g_y] = min_Y

                elseif genes[p, g_y] > max_Y

                    genes[p, g_y] = max_Y
                end
            end
        end
    end

    if cmp(mutation_type, "randreset") == 0

        #mutate what controls X
        mut_array_X_raw = rand(pop_size, 4)
        mut_array_X_ones = ones(size(mut_array_X_raw))
        mut_array_X_raw[mut_array_X_raw .> mutation_rate] .= 0
        mut_array_X_ones[mut_array_X_raw .== 0] .= 0
        mut_array_X = mut_array_X_raw ./ mutation_rate 
        mut_array_X = Int32.(floor.(20*((mut_array_X.*(max_X - min_X)) .+ min_X*mut_array_X_ones)))./20

        #mutate what controls Y
        mut_array_Y_raw = rand(pop_size, 4)
        mut_array_Y_ones = ones(size(mut_array_Y_raw))
        mut_array_Y_raw[mut_array_Y_raw .> mutation_rate] .= 0
        mut_array_Y_ones[mut_array_Y_raw .== 0] .= 0
        mut_array_Y = mut_array_Y_raw ./ mutation_rate 
        mut_array_Y = Int32.(floor.(20*((mut_array_Y.*(max_Y - min_Y)) .+ min_Y*mut_array_Y_ones)))./20
        
        #make all irrelevant things 0
        mut_array_X[2,:] .= 0 
        mut_array_X[4,:] .= 0 
        mut_array_Y[1,:] .= 0 
        mut_array_Y[3,:] .= 0 

        #make the genes which are mutated 0
        destroyer_X = ones(size(mut_array_X))
        destroyer_X[mut_array_X .!= 0] .= 0
        destroyer_Y = ones(size(mut_array_Y))
        destroyer_Y[mut_array_Y .!= 0] .= 0

        #set the values
        genes = genes.*destroyer_X.*destroyer_Y
        
    end

    return genes
end

#generates the creatures at the start of the simulation
function generate_creatures(X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size, num_points)

    fitness = ones(pop_size)

    creatures = []
    p_0 = [X_start_point, Y_start_point]    #start point
    p_1 = [X_end_point, Y_end_point]        #end point

    for p = 1:pop_size

        control_points = [[X_start_point, Y_start_point], [genes[p,1], genes[p,3]], [genes[p,2], genes[p,4]], [X_end_point, Y_end_point]]

        T = range(0, 1, length = num_points)  # Parameter values from 0 to 1
        points = [bezier_curve(control_points, t) for t in T]
        #To avoid Anys in the array
        if p == 1 
            creatures = [points]
        else
            creatures = append!(creatures, [points])
        end
    end
    return creatures, fitness
end

#plots a creature curve to show the creature
function plot_creature_curve(creature)
    points = hcat(creature...)'
    plot(points[:,1], points[:,2])
    gui()
    readline()
end

#kill a few creatures
function kill_creatures(creatures, fitness, genes, kill_rate, pop_size)

    rand_upper_limit = 1.5 #maximum multiple by which a creature's likelihood to die can be increased
    rand_lower_limit = 0.5 #maximum multiple by which a creature's likelihood to die can be decreased

    maximum_creatures = (1 - kill_rate)*pop_size
    kill_count = 0

    live_locs = []
    k = 0

    #find the locations at which to kill creatures
    while pop_size > maximum_creatures
        for p = 1:pop_size
            kill_likelihood = (maximum(fitness) - fitness[p])/maximum(fitness)
            adjusted_kill_likelihood = randfloat(0.5, 1.5)*kill_likelihood*kill_rate  #to make the likelihood of dying slightly luck based
            if rand() < adjusted_kill_likelihood
                pop_size = pop_size - 1
                kill_count = kill_count + 1
            else
                if k == 0 
                    live_locs = [p]
                else
                    live_locs = append!(live_locs, p)
                end
                k = 1
            end
        end
    end

    #kill the creatures
    creatures_new = []
    k = 0
    for i = 1:pop_size
        if i in live_locs
            if k == 0 
                creatures_new = creatures[i]
            else
                creatures_new = append!(creatures_new, creatures[i])
            end
            k = 1
        end
    end
    creatures = creatures_new

    #tell the user how many creatures have been killed
    return creatures, fitness, genes, pop_size, kill_count
end

#function to generate parent pairs
function select_parents(creatures, fitness, genes, pop_size, selection_type, num_parents)

    parents = []
    for par = 1:num_parents
        if cmp(selection_type, "random") == 0 
            parentloc1 = rand(1:pop_size)
            parentloc2 = rand(1:pop_size) 
            if par == 1
                parents = [[parentloc1, parentloc2]]
            else
                parents = append!(parents, [parentloc1, parentloc2])
            end
        end

        
        
        if cmp(selection_type, "roulette") == 0

            parentloc1 = 0
            parentloc2 = 0

            fitness = fitness .- minimum(fitness)
            sum_fitness = sum(fitness)

            #parent 1
            spinner = randfloat(0, sum_fitness)
            partial_sum = 0

            for c = 1:pop_size
                partial_sum = partial_sum + fitness[c]
                if partial_sum > spinner
                    parenloc1 = c
                end
            end

            #parent 2
            spinner = randfloat(0, sum_fitness)
            partial_sum = 0

            for c = 1:pop_size
                partial_sum = partial_sum + fitness[c]
                if partial_sum > spinner
                    parenloc2 = c
                    break
                end
                if c == pop_size
                    c = 1
                end
            end
            
            #set parents
            if par == 1
                parents = [[parentloc1, parentloc2]]
            else
                parents = vcat(parents, [[parentloc1, parentloc2]])
            end
        end

        if cmp(selection_type, "tournament") == 0

            parentloc1 = 0
            parentloc2 = 0

            #parent 1
            competitors = sample(1:pop_size, 3, replace = false)
            max_val, max_idx = findmax(fitness[competitors])
            parentloc1 = competitors[max_idx]

            #parent 2
            competitors = sample(1:pop_size, 3, replace = false)
            max_val, max_idx = findmax(fitness[competitors])
            parentloc2 = competitors[max_idx]

            if par == 1
                parents = [[parentloc1, parentloc2]]
            else
                parents = vcat(parents, [[parentloc1, parentloc2]])
            end
        end
    end
    return parents
end

#function to perform the crossover operation
function crossover(parent_genes, crossover_type)
    #takes in two genes, and returns two genes after appropriate crossover operation
    new_genes = zeros(2,4)

    #this mode splices genes in half and exchanges them
    if cmp(crossover_type, "onepoint_half") == 0

        #find midpoints of each of the genes
        print(parent_genes[1,:])
        readline()
        midpoint1 = div(length(parent_genes[1,:]), 2)
        midpoint2 = div(length(parent_genes[2,:]), 2)

        #split the genes into 4 quarters
        quarter_1 = parent_genes[1, 1:midpoint1]
        quarter_2 = parent_genes[1, (midpoint1+1):end]
        quarter_3 = parent_genes[2, 1:midpoint2]
        quarter_4 = parent_genes[2, (midpoint2+1):end]
        
        #exchange the quarters of the genes
        
        new_genes[1,:] = vcat( quarter_1, quarter_4)    
        new_genes[2,:] = vcat( quarter_3, quarter_2)

    end

    #performs uniform crossover at a random point, works only for equal sized genes
    if cmp(crossover_type, "onepoint_random") == 0

        #find midpoints of each of the genes
        point1 = rand(1:(length(parent_genes[1,:])-1))
        point2 = point1+1

        #if the points are neither the end nor the start
        if point2 != length(parent_genes[1,:]) && point1 != 1
            #split the genes into 4 quarters
            quarter_1 = parent_genes[1, 1:point1]
            quarter_2 = parent_genes[1, point2:end]
            quarter_3 = parent_genes[2, 1:point1]
            quarter_4 = parent_genes[2, point2:end]

            #exchange the quarters of the genes
            new_genes[1, :] = vcat( quarter_1, quarter_4)    
            new_genes[2, :] = vcat( quarter_3, quarter_2)

        #if point 2 is the end
        elseif point2 == length(parent_genes[1, :])
            #split the genes into 4 quarters
            quarter_1 = parent_genes[1, 1:point1]
            quarter_2 = parent_genes[1, point2]
            quarter_3 = parent_genes[2, 1:point1]
            quarter_4 = parent_genes[2, point2]

            #exchange the quarters of the genes
            new_genes[1, :] = vcat( quarter_1, quarter_4)    
            new_genes[2, :] = vcat( quarter_3, quarter_2)
        
        #if point1 is the start
        elseif point1 == 1
            #split the genes into 4 quarters
            quarter_1 = parent_genes[1, 1]
            quarter_2 = parent_genes[1, point2:end]
            quarter_3 = parent_genes[2, 1]
            quarter_4 = parent_genes[2, point2:end]

            #exchange the quarters of the genes
            new_genes[1, :] = vcat( quarter_1, quarter_4)    
            new_genes[2, :] = vcat( quarter_3, quarter_2)
        end

    end

    #switches gene strings at multiple different points
    if cmp(crossover_type, "multipoint_random") == 0
        print("This functionality is not implemented yet.\n")
    end

    #selects a random gene for the first offspring between the parents and gives the complement to the other offspring.
    if cmp(crossover_type, "uniform_random") == 0
        for g = 1:length(parent_genes[1, :])
            allele_first, allele_second = sample(1:2, 2, replace = false)
            new_genes[1, g] = parent_genes[g, allele_first]
            new_genes[2, g] = parent_genes[g, allele_second]
        end
    end
    return new_genes
end

#calculates the length of a creature to find material wastage
function calculate_creature_length(creature)
    points = hcat(creature...)'
    X = points[:,1]
    Y = points[:,2]
    distance = 0
    for p = 1:(length(X)-1)
        distance = distance + sqrt((X[p+1] - X[p])^2 + (Y[p+1] - Y[p])^2)
    end

    return distance
end

function generate_domain(creature, X_len_real, Y_len_real, X_lat_d, Y_lat_d, X_tube_start, Y_tube_start_top, symmetry_L_Y)

    dx = 1/X_lat_d  #length scale conversion factor
    dy = dx
    Nx = Int32(floor(X_len_real * X_lat_d)) #Number of X lattice points
    Ny = Int32(floor(Y_len_real * Y_lat_d)) #Number of Y lattice points
    SB = zeros(Ny,Nx) #solid body array
    FB = ones(Ny,Nx)  #fluid body array
    points_inlet = hcat(creature...)' #creature inlet points
    X_inlet = points_inlet[:,1] #X points
    Y_top_inlet = points_inlet[:,2] #Y points
    Y_bottom_inlet = -Y_top_inlet .+ 2*symmetry_L_Y

    #reflect tube about mirror line
    Y_tube_start_bottom = -Y_tube_start_top + 2*symmetry_L_Y


    #make the instrument tube
    Ny_tube_start_top = Int32(floor(Y_tube_start_top*Y_lat_d)) #Y index of cell where the tube starts, top
    Ny_tube_start_bottom = Int32(floor(Y_tube_start_bottom*Y_lat_d)) #Y index of cell where the tube starts, bottom
    Nx_tube_start = Int32(floor(X_tube_start*X_lat_d)) #X index of cell where the tube starts

    #top of tube
    SB[Ny_tube_start_top:Ny, Nx_tube_start:Nx] .= 1
    FB[Ny_tube_start_top:Ny, Nx_tube_start:Nx] .= 0

    #bottom of tube
    SB[1:Ny_tube_start_bottom, Nx_tube_start:Nx] .= 1
    FB[1:Ny_tube_start_bottom, Nx_tube_start:Nx] .= 0

    #make the inlet
    Ny_inlet_top = Int32.(floor.(Y_top_inlet.*Y_lat_d)) #cell locations of top inlet surface
    Ny_center = Int32.(floor.(Ny/2))
    Ny_inlet_bottom = Int32.(floor.(Y_bottom_inlet.*Y_lat_d)) #cell locations of bottom inlet surface

    for i = 1:Nx_tube_start

        #top surface
        SB[Ny_inlet_top[i]:Ny, i] .= 1
        FB[Ny_inlet_top[i]:Ny, i] .= 0

        #bottom surface
        SB[1:Ny_inlet_bottom[i], i] .= 1
        FB[1:Ny_inlet_bottom[i], i] .= 0

        #make sure there is a path for the fluid
        SB[Ny_center:(Ny_center+10), i] .= 0
        SB[(Ny_center-10):(Ny_center), i] .= 0
        FB[Ny_center:(Ny_center+10), i] .= 1
        FB[(Ny_center-10):(Ny_center), i] .= 1
    end

    

    return SB, FB
    
end

function Vikasa()

    #global variables
    show_creatures = false
    mutate_before_start = true
    save = true
    save_frequency = 1


    #optimization variables
    global_pop_size = 10                       #population size every generation
    pop_size = global_pop_size                 #population size within generations 
    kill_rate = 0.2                            #number of creatures to kill
    mutation_rate = 0.1                        #mutation rate every generation
    number_gen = 1000                           #number of generations, effectively functions as the number of iterations
    mutation_value = 0.05                       #value to mutate by
    convergence_parameter = 1E-5                #maximum change in fitness between generations to consider solution as converged

    #domain variables
    X_start_point = 0
    Y_start_point = 0.8
    X_end_point = 1
    Y_end_point = 0.6
    X_len_real = 5                                   #maximum range of X
    Y_len_real = 0.8                                 #maximum range of Y
    X_tube_start = X_end_point                       #X location at which tube starts
    Y_tube_start_top = Y_end_point                   #Y location at which tube starts
    symmetry_L_Y = 0.4                               #Y coordinate across which symmetry exists
    X_lat_d = 100                                    #lattice density in X direction
    Y_lat_d = 100                                    #lattice density in Y direction
    Nx = Int32(floor(X_len_real * X_lat_d))          #Number of X lattice points
    Ny = Int32(floor(Y_len_real * Y_lat_d))          #Number of Y lattice points
    SB = zeros(Ny,Nx)
    SBs = [SB, SB, SB, SB, SB, SB, SB, SB, SB, SB]   #solid body meshes
    FBs = [SB, SB, SB, SB, SB, SB, SB, SB, SB, SB]   #fluid body meshes

    #Simulation parameters 
    t = 200
    Nt = 20000
    tau = 0.61
    Re = 50
    t_array = [1.0,0.0,0.0,0.0,0.0]
    kill_count = 0


    print("[VIKASA 0.1] - Genetic algorithms for flow applications\n")
    print("creating genes...\n")
    genes = zeros(pop_size, 4)
    genes = generate_population(X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size)
    print("creating creatures...\n")
    creatures, fitness = generate_creatures(X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size, Int32(floor(X_tube_start*X_lat_d)))
    



    
    if mutate_before_start
        print("mutating creatures...\n")
        genes = mutate(mutation_rate, mutation_value, X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size, "randreset")
    end

    for g = 1:number_gen

        #generate the domains
        for i = 1:pop_size
            SBs[i], FBs[i] = generate_domain(creatures[i], X_len_real, Y_len_real, X_lat_d, Y_lat_d, X_tube_start, Y_tube_start_top, symmetry_L_Y)
        end

        #show the creatures
        if show_creatures
            print("showing creatures...\n")
            for p = 1:pop_size
                if p != pop_size
                    print("press any key to view the next creature.[creature ", p, " of ", pop_size,"]\n")
                else
                    print("press any key to exit.[creature ", p, " of ", pop_size,"]\n")
                end
                plot_creature_curve(creatures[p])
                heatmap(SBs[p], ylimits = (0,Ny), xlimits = (0,Nx), aspect_ratio = :equal, c= :ice)
                gui()
                readline()
            end
        end

        t = mean(t_array[t_array .> 0])[1]
        ttemp = CUDA.@elapsed begin
        
        days = Int32(floor(t*(number_gen-g)/86400))
        hours = Int32(floor(t*(number_gen-g)/3600) - 24*days)
        minutes = Int32(floor(t*(number_gen-g)/60) - 60*hours - 24*60*days) #calculate time in days, hours & minutes

        

        for i = 1:pop_size
            creature_length = 2 - calculate_creature_length(creatures[i]) #calculate creature length
            print("Running ArvanFlow for creature ", i, " in generation ", g, ".\n")
            TKE, TI, error = ArvanFlow(X_len_real, Y_len_real, X_lat_d, SBs[i], FBs[i], "generation-"*string(g)*"-creature-"*string(i), t, Nt, tau, Re) #use arvanflow to calculate turbulent intensity
            fitness[i] = (1/TI)*0.8E-4 + creature_length*0.2 #calculate fitness

            if error 
                fitness[i] = 1E-5
                print("This creature is no good! Skipping over to the next.")
            end
        end

        

        average_fitness = mean(fitness)
        average_fitness_frmt = @sprintf "%.3f" average_fitness
        maximum_fitness = maximum(fitness)
        maximum_fitness_frmt = @sprintf "%.3f" maximum_fitness

        if minutes == 0 #make minutes 1 just before simulaation finishes
            minutes = 1
        end
        if g%10 == 1
            print("\nIteration \tTime \t \tAverage Fitness \tMaximum Fitness \tDeaths \n \n") #show headings every 10 iterations
        end
        print(Int32(g), "\t \t", days, ":", hours, ":", minutes, "\t \t", average_fitness_frmt, "\t \t \t", maximum_fitness_frmt, "\t \t \t", kill_count, "\n") # display values

        if abs((average_fitness - maximum_fitness)/(maximum_fitness)) < convergence_parameter
            print("solution has converged at fitness = ", maximum_fitness_frmt, " in ", g, " generations.\n")
            readline()
            exit()
        end

        creatures, fitness, genes, pop_size, kill_count = kill_creatures(creatures, fitness, genes, kill_rate, pop_size)
        parents = select_parents(creatures, fitness, genes, pop_size, "tournament", Int32(floor(global_pop_size/2)))
        new_genes = zeros(global_pop_size, 4)
        k = 0
        filled = []
        for parent_couple in parents
            
            parent1genes = genes[parent_couple[1],:]
            parent2genes = genes[parent_couple[2],:]
            parent_genes = zeros(2, 4)
            parent_genes[1,:] = parent1genes
            parent_genes[2,:] = parent2genes
            children = crossover(parent_genes, "onepoint_random")

            new_genes[parent_couple[1],:] = children[1,:]
            new_genes[parent_couple[2],:] = children[2,:]
            if k == 0
                filled = [parent_couple[1], parent_couple[2]]
            else
                filled = append!(filled, parent_couple[1])
                filled = append!(filled, parent_couple[2])
            end
            k = 1
        end
        if pop_size < global_pop_size
            for i = 1:global_pop_size
                if mean(new_genes[i,:]) == 0
                    new_genes[i,:] = new_genes[rand(filled),:]
                end
            end
        end
        genes = new_genes
        pop_size = global_pop_size
        genes = mutate(mutation_rate*0.1, mutation_value, X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size, "randreset")
        genes = mutate(mutation_rate, mutation_value, X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size, "addsub")
        creatures, fitness = generate_creatures(X_start_point, Y_start_point, X_end_point, Y_end_point, genes, pop_size, Int32(floor(X_tube_start*X_lat_d)))

        if save && (g%save_frequency == 0)
            print("Saving...\n")
            num_digits_max = ndigits(number_gen)
            num_digits_iters = ndigits(g)
            format_it = string(g)
            format_it = lpad(format_it, num_digits_max - num_digits_iters, '0')
            save_object(saved_filename*"-creatures-"*format_it*".jld2" , Array(creatures))
            save_object(saved_filename*"-fitness-"*format_it*".jld2" , Array(fitness))
            print("Saved!\n")
        end


        end
        t_array = circshift(t_array, 1)
        t_array[1] = ttemp
        print("Optimization complete!\n")
        if save
            print("Writing final files...\n")
            save_object("creatures-endwrite.jld2" , Array(creatures))
            save_object("fitness-endwrite.jld2" , Array(fitness))
            print("Saved at the end!\n")
        end
        
    end
end
Vikasa()