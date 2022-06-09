-- Gkyl -----------------------------------------------------------------------
-- Z.Liu 2/10/2021
-- This input file is for linear benchmarking of Ion acoustic instability

local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell()

-- electron parameters
vDriftElc = 0.01   ---modified from 0.159
vtElc = 0.02
-- ion parameters
vDriftIon = 0.0
vtIon = 0.000566      ---- modified from 0.001 (use Te = 50 Ti)
-- mass ratio
massRatio = 25.0  ----modified from 25 

knumber = 10.0 -- wave-number
perturbation = 1.0e-4 -- distribution function perturbation

lx = {1.0}
nx = {64}

local function maxwellian1v(v, vDrift, vt)
   return 1/math.sqrt(2*math.pi*vt^2)*math.exp(-(v-vDrift)^2/(2*vt^2))
end

local function sponEmissionSource(x_table, t, lx_table, ncells_table, p)
    -- x_table = {x, y} are independent variables.
    -- t is the time
    -- lx_table = {xmax-xmin, ymax - ymin}
    -- ncells_table = {ncellsx, ncellsy}
    -- p is polynomial order.
    ------------------------
    -- returns stochastic source term (\delta E_x, \delta E_y, delta f_i, \delta f_e) that models spontaneous emission of ion-acoustic waves
    local Nx = ncells_table[1]*3 -- Number of spatial degrees of freedom along x
    local Lx = lx_table[1]
    local x, vx = x_table[1], x_table[2]
    local fIon = 0.0
    local ksquared = 0.0
    math.randomseed(math.floor(1000000*t)) --I want all xs and ys (and vx, vy) to see the same random phase at a given time step.  Since t will be a fraction, need to multiply it so that we don't get the same random number thousands of times.
    for nx = -math.floor(Nx/2), math.floor(Nx/2) do   --need to divied by two because we are including nx and -nx by using sines and cosines
        ksquared = math.pow(2*nx*math.pi /Lx,2)
        if ksquared > 0.0 then
            fIon = fIon + math.cos(2*math.pi*(nx*x/Lx + math.random() ))
        end  
    end
    return {fIon}
 end

plasmaApp = Plasma.App {
   logToFile = true,

   tEnd =1600.0, -- end time
   nFrame = 800, -- number of output frames

   lower = {0.0}, -- configuration space lower left
   upper = {1.0}, -- configuration space upper right
   cells = {64}, -- configuration space cells
   basis = "serendipity", -- one of "serendipity" or "maximal-order"
   polyOrder = 2, -- polynomial order
   timeStepper = "rk3", -- one of "rk2" or "rk3"
   cflFrac = 0.9,

   -- decomposition for configuration space
   decompCuts = {8}, -- cuts in each configuration direction
   useShared = false, -- if to use shared memory

   -- boundary conditions for configuration space
   periodicDirs = {1}, -- periodic directions
   -- integrated moment flag, compute quantities 1000 times in simulation
   calcIntQuantEvery = 0.001,   

   -- electrons
   elc = Plasma.Species {
      charge = -1.0, mass = 1.0,
      -- velocity space grid
      lower = {-6.0*vtElc},
      upper = {6.0*vtElc},
      cells = {256},
      decompCuts = {1},
      -- initial conditions
      init = function (t, xn)
       local x, v = xn[1], xn[2]
        local fv = maxwellian1v(v, vDriftElc, vtElc)
	 return fv*(1)
      end,
      evolve = true, -- evolve species?

      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
   },

   -- ions
   ion = Plasma.Species {
      charge = 1.0, mass = massRatio,
      -- velocity space grid
      lower = {-16.0*vtIon},   
      upper = {16.0*vtIon},
      cells = {48},
      decompCuts = {1},
      -- initial conditions
      init = function (t, xn)
       local x, v = xn[1], xn[2]
        local fv = maxwellian1v(v, vDriftIon, vtIon)
        local dfv = sponEmissionSource({x,v},t, lx, nx, polyOrder)[1]* maxwellian1v(v, vDriftIon, vtIon)
	  return fv + perturbation*dfv
      end,
      evolve = true, -- evolve species?

      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
   },   

   -- field solver
   field = Plasma.Field {
      epsilon0 = 1.0,
      evolve = true, -- evolve field?
      hasMagneticField = false,
   },
}
-- run application
plasmaApp:run()
