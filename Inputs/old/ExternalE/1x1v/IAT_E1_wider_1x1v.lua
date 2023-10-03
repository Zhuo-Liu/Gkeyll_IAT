-- Gkyl -----------------------------------------------------------------------
-- Z.Liu 5/6/2021
-- wider electron velocity range
-- all modes excited

local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell()

-- Electron parameters.
vDriftElc = 0.00  -- Modified from 0.159.
vtElc     = 0.02
-- Ion parameters.
vDriftIon = 0.0
vtIon     = 0.000566      -- Modified from 0.001 (use Te = 50 Ti).
-- Mass ratio.
massRatio = 25.0  -- Modified from 25.

knumber      = 10.0    -- Wave-number.
perturbation = 1.0e-4  -- Distribution function perturbation.
noise = 1.0e-6

-- Collision frequencies.
nuee = 0.0001
nuei = 0.0 --nuee              --RLW 
nuii = nuee/math.sqrt(massRatio)  --should have been nuee/math.sqrt(massRatio)*math.pow(tempRatio,1.5)
nuie = 0.0 -- nuee/massRatio

local function maxwellian1v(v, vDrift, vt)
   return 1/math.sqrt(2*math.pi*vt^2)*math.exp(-(v-vDrift)^2/(2*vt^2))
end

local function sponEmissionSource(x_table, t, lx_table, ncells_table, p)
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

   tEnd        = 2000.0,         -- End time.
   nFrame      = 1000,           -- Number of output frames.

   lower       = {0.0},         -- Configuration space lower left.
   upper       = {1.0},         -- Configuration space upper right.
   cells       = {64},         -- Configuration space cells.
   basis       = "serendipity", -- One of "serendipity" or "maximal-order".
   polyOrder   = 2,             -- Polynomial order.
   timeStepper = "rk3",         -- One of "rk2" or "rk3".
   cflFrac     = 0.9,

   -- Eecomposition for configuration space
   decompCuts = {16},  -- Cuts in each configuration direction.
   useShared  = false, -- If to use shared memory.

   -- Boundary conditions for configuration space.
   periodicDirs = {1}, -- Periodic directions.
   -- Integrated moment flag, compute quantities 1000 times in simulation.
   calcIntQuantEvery = 0.001,   

   -- Electrons.
   elc = Plasma.Species {
      charge = -1.0, mass = 1.0,
      -- Velocity space grid.
      lower      = {-3.0*vtElc},
      upper      = { 12.0*vtElc},
      cells      = {120},
      decompCuts = {1},
      -- Initial conditions.
      init = function (t, xn)
         local x, v = xn[1], xn[2]
         local fv = maxwellian1v(v, vDriftElc, vtElc)
	 return fv*(1)
      end,
      evolve = true, -- Evolve species?

      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
      coll = Plasma.LBOCollisions{
         collideWith = {'elc', 'ion'},
        frequencies = {nuee, nuei},
       }
   },

   -- Ions.
   ion = Plasma.Species {
      charge = 1.0, mass = massRatio,
      -- Velocity space grid.
      lower      = {-16.0*vtIon},   
      upper      = { 16.0*vtIon},
      cells      = {32},
      decompCuts = {1},
      -- Initial conditions.
      init = function (t, xn)
         local x, v = xn[1], xn[2]
         local fv = maxwellian1v(v, vDriftIon, vtIon)
         local dfv = sponEmissionSource({x,v},t, lx, nx, polyOrder)[1]* maxwellian1v(v, vDriftIon, vtIon)
         return fv + perturbation*dfv
      end,
      source = function (t,xn)
         local x, v = xn[1], xn[2]
         local dfv = sponEmissionSource({x,v},t, lx, nx, polyOrder)[1]*maxwellian2v(v, vDriftIon, vtIo) 
         return noise*dfv
      end,

      evolve = true, -- Evolve species?      
      diagnosticMoments = { "M0", "M1i"},
      diagnosticIntegratedMoments = { "intM0", "intM1i","intM2Flow", "intM2Thermal",},

      coll = Plasma.LBOCollisions{
         collideWith = {'elc', 'ion'},
         frequencies = {nuie, nuii},
      }
   },   

   -- Field solver.
   field = Plasma.Field {
      epsilon0 = 1.0,
      evolve   = true, -- Evolve field?
      hasMagneticField = false,
   },

   externalField = Plasma.ExternalField {
      hasMagneticField = false,
      emFunc = function(t, xn)
         local extE_x, extE_y, extE_z = -0.00001, 0., 0.
         return extE_x, extE_y, extE_z
      end,
      evolve = false, -- Evolve field?
   },
}
-- Run application.
plasmaApp:run()
