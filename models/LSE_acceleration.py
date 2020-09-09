import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LeastSquareEstimation(nn.Module):
    def __init__(self):
        super(LeastSquareEstimation, self).__init__()
	
    def forward(self, A, b):
        '''
          solve Ax = b
          ==> A'Ax = A'b
          ==> x = (A'A)^(-1)@A'@b
          
          input: A [M, N]    b [M, 1]
          
        '''
        A_trans = A.t()
        out = torch.matmul(A_trans, A)    #  A'A
        out = torch.inverse(out)          # (A'A)^(-1)
        out = torch.matmul(out, A_trans)  # (A'A)^(-1)@A'
        
        out = torch.matmul(out, b)  # (A'A)^(-1)@A'@b
        
        inver_result = torch.matmul(A,out)
        error = F.mse_loss(inver_result, b, reduction='sum')
        
        result = out
        return result, error


class LSE_acceleration_solver_cpu(nn.Module):
    '''
          solve Ax = b
          ==> A'Ax = A'b
          ==> x = (A'A)^(-1)@A'@b
    '''
    def __init__(self):
        super(LSE_acceleration_solver, self).__init__()
        A = np.array([-1, 0.5, 1 ,0.5, 2, 2]).reshape((3, 2))   # coefficient matrix
        A_tensor = torch.from_numpy(A).float()
        A_trans = A_tensor.t()
        cof = torch.matmul(A_trans, A_tensor)    #  A'A
        cof = torch.inverse(cof)          # (A'A)^(-1)
        self.cof = torch.matmul(cof, A_trans)  # (A'A)^(-1)@A'
        self.A = A_tensor
	
    def forward(self, f1, f2, f3):
        '''
          input: (channel=1)
            - f1 [B, H, W]
            - f2 [B, H, W]
            - f3 [B, H, W]
            
          return:
            - acc [B, H, W]
            - v0 [B, H, W]
          
        '''
        with torch.no_grad():
            B, H, W = f1.size()
            cof = self.cof.repeat(B, 1, 1) # [B, 2, 3]
            
            B, H, W = f1.size()
            f1 = f1.view(B, -1)  # [B, HW]
            f2 = f2.view(B, -1)  # [B, HW]
            f3 = f3.view(B, -1)  # [B, HW]
            b = torch.stack([f1, f2, f3], dim=1)   # # [B, 3, HW]
            
            result = torch.bmm(cof.cpu(), b.cpu())   # [B, 2, HW]
            
            invers_result = torch.bmm(self.A.repeat(B,1,1), result.cpu())
            error = F.mse_loss(invers_result, b.cpu(), reduction='none')
            error = error.view(B, 3, H, W)
            error = torch.mean(error, dim=1, keepdim=False)  # [B, H, W]
            
            result_v0 = result.view(B, 2, H, W)[:,0,:,:]    # [B, H, W]
            result_acc = result.view(B, 2, H, W)[:,1,:,:]   # [B, H, W]
               
        return result_v0.cuda(), result_acc.cuda(), error.cuda()
  

class LSE_acceleration_solver(nn.Module):
    '''
          solve Ax = b
          ==> A'Ax = A'b
          ==> x = (A'A)^(-1)@A'@b
    '''
    def __init__(self):
        super(LSE_acceleration_solver, self).__init__()
        print('Using LSE_acceleration_solver: GPU')
        A = np.array([-1, 0.5, 1 ,0.5, 2, 2]).reshape((3, 2))   # coefficient matrix
        A_tensor = torch.from_numpy(A).float().to(torch.device("cuda:0"))
        A_trans = A_tensor.t()
        cof = torch.matmul(A_trans, A_tensor)    #  A'A
        cof = torch.inverse(cof)          # (A'A)^(-1)
        self.cof = torch.matmul(cof, A_trans)  # (A'A)^(-1)@A'
        self.A = A_tensor
	
    def forward(self, f1, f2, f3):
        '''
          input: (channel=1)
            - f1 [B, H, W]
            - f2 [B, H, W]
            - f3 [B, H, W]
            
          return:
            - acc [B, H, W]
            - v0 [B, H, W]
          
        '''
        with torch.no_grad():
            B, H, W = f1.size()
            cof = self.cof.repeat(B, 1, 1) # [B, 2, 3]
            
            B, H, W = f1.size()
            f1 = f1.view(B, -1)  # [B, HW]
            f2 = f2.view(B, -1)  # [B, HW]
            f3 = f3.view(B, -1)  # [B, HW]
            b = torch.stack([f1, f2, f3], dim=1).to(torch.device("cuda:0"))   # # [B, 3, HW]
            
            result = torch.bmm(cof, b)   # [B, 2, HW]           
            invers_result = torch.bmm(self.A.repeat(B,1,1), result)
            
            result_v0 = result.view(B, 2, H, W)[:,0,:,:]    # [B, H, W]
            result_acc = result.view(B, 2, H, W)[:,1,:,:]   # [B, H, W]
               
        return result_v0.cuda(), result_acc.cuda() #, error.cuda()

     
class compute_acceleration(nn.Module):
    def __init__(self, model):
        super(compute_acceleration, self).__init__()
        print('Using LSE acceleration!')
        self.flownet = model
        self.LSE_acc_solver = LSE_acceleration_solver()
        
	
    def forward(self, Ia, Ib, Ic, Id, t):
        '''
          
          
        '''
        with torch.no_grad():
            # F(0-->t)
            F_ba = self.flownet(Ib, Ia).float()  # [B, 2, H, W]
            F_bc = self.flownet(Ib, Ic).float()       
            F_bd = self.flownet(Ib, Id).float()
            
            F_ba_u = F_ba[:,0,:,:]    # [B, H, W]
            F_ba_v = F_ba[:,1,:,:]    # [B, H, W]
            
            F_bc_u = F_bc[:,0,:,:]
            F_bc_v = F_bc[:,1,:,:]
            
            F_bd_u = F_bd[:,0,:,:]
            F_bd_v = F_bd[:,1,:,:]
            
            result_v0_u, result_acc_u = self.LSE_acc_solver(F_ba_u, F_bc_u, F_bd_u)  # result_v0_H [B, H, W]
            result_v0_v, result_acc_v = self.LSE_acc_solver(F_ba_v, F_bc_v, F_bd_v)  # result_v0_W [B, H, W]
            
            acc = torch.stack([result_acc_u, result_acc_v], dim=1)    # [B, 2, H, W]
            v0 = torch.stack([result_v0_u, result_v0_v], dim=1)
                              
            # simple Quadratic
            acc_1 = F_ba + F_bc
            acc_2 = (2/3)*F_ba + (1/3)*F_bd
            acc_3 = F_bd - 2*F_bc
            v0_1 = 0.5*(F_bc - F_ba)            
            
            
            x = acc_1-acc_2            
            sym = 5
            fac = 1
            
            alpha = -0.5 * (torch.exp(fac*(torch.abs(x)-sym)) - torch.exp(-fac*(torch.abs(x)-sym))) / (torch.exp(fac*(torch.abs(x)-sym))  +  torch.exp(-fac*(torch.abs(x)-sym))) + 0.5
            acc_final = torch.where((acc_1*acc_2>0)&(acc_1*acc_3>0), alpha*acc + (1-alpha)*acc_1, acc_1)  
            v0_final = torch.where((acc_1*acc_2>0)&(acc_1*acc_3>0), alpha*v0 + (1-alpha)*v0_1, v0_1)
                    
            
            F_bt = v0_final*t + 0.5*acc_final*t**2
       
        return F_bt
        

if __name__ == '__main__':       
    LSE = LeastSquareEstimation()
    
    A = np.array([-1, 0.5, 1 ,0.5, 2, 2]).reshape((3, 2))
    b = np.array([-1.5, 2.5, 6])
    
    A_tensor = torch.from_numpy(A).float()   # [M, N]
    b_tensor = torch.from_numpy(b).float()
    
    b_tensor = b_tensor.view(-1,1)   # [M, 1]
    print(b_tensor.size())
    print(b_tensor)
    
    
    b_tensor = b_tensor.repeat(1,4).unsqueeze(0) # [B, 3, MN]
    print(b_tensor.size())
    print(b_tensor)
    
    
    
    #result, error = LSE(A_tensor, b_tensor)
    #print(result.size())
    #print(result)
    #print(error.size())
    #print(error)
    
    A_trans = A_tensor.t()
    out = torch.matmul(A_trans, A_tensor)    #  A'A
    out = torch.inverse(out)          # (A'A)^(-1)
    out = torch.matmul(out, A_trans)  # (A'A)^(-1)@A'
    out = out.unsqueeze(0) # [b, 2, 3]
    
    
    result = torch.bmm(out, b_tensor)
    print(result.size())
    print(result)
    
    
    print('##############################')
    f1 = np.array([-2, -6, -3, -6]).reshape((2, 2))
    f2 = np.array([4, 4, 4, 4]).reshape((2, 2))
    f3 = np.array([10, 6, 10, 7]).reshape((2, 2))
    
    # input
    f1 = torch.from_numpy(f1).unsqueeze(0).float()  # [B, M, N]
    f2 = torch.from_numpy(f2).unsqueeze(0).float()
    f3 = torch.from_numpy(f3).unsqueeze(0).float()
    
    lse_acc = LSE_acceleration_solver()
    result_v0, result_acc, error = lse_acc(f1, f2, f3)
    print(result_v0)
    print(result_acc)
    print(error)
    



