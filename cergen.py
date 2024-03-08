import random
import math
# import numpy as np
from typing import Union

def cekirdek(sayi: int):
    """
    Sets the seed for random number generation
    """
    random.seed(sayi)

def generate_nested_list_dogal(shape, aralik, depth=0):
    """
    Generates a nested list of integers with the specified shape and range.
    """
    if depth == len(shape) - 1:
        return [random.randint(aralik[0], aralik[1]) for _ in range(shape[depth])]
    else:
        return [generate_nested_list_dogal(shape, aralik, depth + 1) for _ in range(shape[depth])]

def rastgele_dogal(boyut, aralik=(0,100), dagilim='uniform'):
    """
    Generates data of specified dimensions with random integer values and returns a gergen object.

    Parameters:
    boyut (tuple): Shape of the desired data.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0,100), which implies a default range.
    dagilim (string, optional): Distribution of random values ('uniform'). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random integer values.
    """
    if dagilim != 'uniform':
        raise ValueError('Invalid distribution.')
    
    if boyut == ():
        return gergen(random.randint(aralik[0], aralik[1]))
    
    veri = generate_nested_list_dogal(boyut, aralik)
    return gergen(veri)

def generate_nested_list_gercek(shape, aralik, depth=0):
    """
    Generates a nested list of floating-point numbers with the specified shape and range.
    """
    if depth == len(shape) - 1:
        return [random.random() * (aralik[1] - aralik[0]) + aralik[0] for _ in range(shape[depth])]
    else:
        return [generate_nested_list_gercek(shape, aralik, depth + 1) for _ in range(shape[depth])]

def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    """
    Generates a gergen of specified dimensions with random floating-point values.

    Parameters:
    boyut (tuple): Shape of the desired gergen.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0.0, 1.0) for uniform distribution.
    dagilim (string, optional): Distribution of random value ('uniform'). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random floating-point values.
    """
    if dagilim != 'uniform':
        raise ValueError('Invalid distribution.')
    
    if boyut == ():
        return gergen(random.random() * (aralik[1] - aralik[0]) + aralik[0])
    
    veri = generate_nested_list_gercek(boyut, aralik)
    return gergen(veri)

class Operation:
    def __call__(self, *operands):
        """
        Makes an instance of the Operation class callable.
        Stores operands and initializes outputs to None.
        Invokes the forward pass of the operation with given operands.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the forward pass of the operation.
        """
        self.operands = operands
        self.outputs = None
        return self.ileri(*operands)

    def ileri(self, *operands):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError
    
class gergen:
    __veri = None # A nested list of numbers representing the data
    D = None # Transpose of data
    __boyut = None # Dimensions of the derivative (Shape)

    def __init__(self, veri=None):
        """
        The constructor for the 'gergen' class.
        
        This method initializes a new instance of a gergen object. The gergen can be
        initialized with data if provided; otherwise, it defaults to None, representing
        an empty tensor.
        
        Parameters:
        veri (int/float, list, list of lists, optional): A nested list of numbers that represents the
        gergen data. The outer list contains rows, and each inner list contains the
        elements of each row. If 'veri' is None, the tensor is initialized without data.
        
        Example:
        To create a tensor with data, pass a nested list:
        tensor = gergen([[1, 2, 3], [4, 5, 6]])
        
        To create an empty tensor, simply instantiate the class without arguments:
        empty_tensor = gergen()
        """
        self.__veri = self.__validate_veri(veri)
        self.__boyut = self.__calculate_boyut(veri)
        self.D = None  # Placeholder for transpose, which needs its own implementation ###################
        # print("veri: ", self.__veri)
        # print("boyut: ", self.__boyut)

    def __validate_veri(self, veri):
        """
        Recursively checks if the input data (veri) is valid (i.e., all sublists are of equal length).
        Returns the validated data or raises a ValueError if the data is invalid.
        """
        if veri is None:
            return None
        elif isinstance(veri, (int, float)):
            return veri
        elif isinstance(veri, list):
            if not veri:
                return veri
            if all(isinstance(item, (int, float)) for item in veri):
                return veri
            elif all(isinstance(item, list) for item in veri):
                length = len(veri[0])
                for sublist in veri:
                    if len(sublist) != length or not all(isinstance(item, (int, float, list)) for item in sublist):
                        raise ValueError('All sublists must be of equal length and contain only numbers or sublists.')
                    self.__validate_veri(sublist)
                return veri
            else:
                raise ValueError('Invalid data type in veri.')
        else:
            raise ValueError('veri must be a number, a list of numbers, or a nested list of numbers.')

    def __calculate_boyut(self, veri):
        """
        Recursively calculates the dimensions (shape) of the tensor based on the input data (veri).
        Returns a tuple representing the dimensions of the tensor.
        """
        if veri is None:
            return None
        elif isinstance(veri, (int, float)):
            return ()
        else:
            boyut = []
            temp = veri
            if isinstance(temp[0], list):
                while isinstance(temp[0], list):
                    boyut.append(len(temp))
                    temp = temp[0]
                boyut.append(len(temp))
            else:
                boyut.append(1)
                boyut.append(len(temp))
            return tuple(boyut) 


    def __getitem__(self, index):
        """
        Enables indexing the gergen, allowing access to specific elements, rows, columns, 
        or sub-tensors using standard Python indexing syntax. This method will be called 
        when an index is used on an instance of the gergen class, such as g[0] for a gergen g.
        """
        if self.__veri is None:
            raise ValueError("Cannot index an empty gergen.")

        if not isinstance(index, tuple):
            index = (index,)  # Make single indexes into a tuple for uniform handling
        
        current = self.__veri
        for idx in index:
            if not isinstance(current, list):
                raise IndexError("Indexing beyond the tensor dimensions")
            if idx < 0:
                idx += len(current)
            if idx < 0 or idx >= len(current):
                raise IndexError("Index out of range")
            current = current[idx]
        return current

    def __str__(self):
        """
        Generates a string representation of the gergen. When
        print(g1) is invoked on a gergen instance g1, this method is automatically called,
        resulting in the display of the tensor’s boyut followed by its veri.
        """
        if self.__veri is None:
            return '[]'
        elif isinstance(self.__veri, (int, float)):
            return f'0 boyutlu skaler gergen:\n{self.__veri}'
        else:
            # Helper function to format nested lists correctly
            def format_nested_list(veri):
                if all(isinstance(item, list) for item in veri):  # Multi-dimensional
                    # Format each row with brackets and join with newline in between
                    formatted_rows = ['[' + ', '.join(map(str, row)) + ']' for row in veri]
                    return '[' + '\n'.join(formatted_rows) + ']'
                else:  # Single-dimensional
                    return '[[' + ', '.join(map(str, veri)) + ']]'

            # Determine if we're dealing with a multi-dimensional tensor or a vector of any size
            boyut = ''
            for i in range(len(self.__boyut)):
                if i != len(self.__boyut) - 1:
                    boyut += f'{self.__boyut[i]}x'
                else:
                    boyut += f'{self.__boyut[i]}'

            # Use the helper function to format the tensor data
            formatted_data = format_nested_list(self.__veri)

            return f'{boyut} boyutlu gergen:\n{formatted_data}'

    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        This method facilitates the multiplication of the gergen either with another gergen instance for
        element-wise multiplication, or with a scalar (int/float), yielding a new gergen object
        as the result. The other parameter is permitted to be a gergen, an integer, or
        a floating-point number. Error handling is incorporated to manage cases where the
        other parameter is neither a gergen object nor a numerical scalar. If the dimensions
        of two gergen instances do not align for element-wise multiplication, or if an
        incompatible type is provided for other, a TypeError or ValueError is raised.
        """
        if self.__veri is None:
            raise ValueError("Cannot multiply an empty gergen.")
        
        if isinstance(other, (int, float)):
            if isinstance(self.__veri, (int, float)):
                # Multiply two scalars
                new_data = self.__veri * other
                return gergen(new_data)
            else:
                # Multiply each element of the gergen of any size by the scalar
                def scalar_mult(data, scalar):
                    if isinstance(data, list):
                        return [scalar_mult(subdata, scalar) for subdata in data]
                    else:
                        return data * scalar
                new_data = scalar_mult(self.__veri, other)
                return gergen(new_data)
        elif isinstance(other, gergen):
            # Element-wise multiplication of two gergen objects
            if self.__boyut != other.__boyut:
                raise ValueError("Cannot multiply gergens with different dimensions.")
            else:
                # Multiply each element of the gergen of any size by the corresponding element of the other gergen
                def elementwise_mult(data1, data2):
                    if isinstance(data1, list) and isinstance(data2, list):
                        return [elementwise_mult(subdata1, subdata2) for subdata1, subdata2 in zip(data1, data2)]
                    else:
                        return data1 * data2
                new_data = elementwise_mult(self.__veri, other.__veri)
                return gergen(new_data)          

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        This method implements division for the gergen, facilitating element-wise division by a scalar
        (an integer or a float), and encapsulates the result in a new gergen instance. True division is 
        employed, ensuring that the result is always a floating-point number, consistent
        with Python 3.x division behavior, even if both operands are integers. Error handling
        mechanism ahould check potential issues: if other is zero, a ZeroDivisionError is
        raised to prevent division by zero. Additionally, if other is not a scalar type (int or
        float), a TypeError is raised to enforce the type requirement for the scalar divisor.
        """
         # Check for division by zero
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        
        # Case where the divisor is a scalar (int or float)
        if isinstance(other, (int, float)):
            # Recursive function for scalar division
            def scalar_div(data, scalar):
                if isinstance(data, list):
                    # If the current item is a list, recur for each item
                    return [scalar_div(subdata, scalar) for subdata in data]
                else:
                    # Base case: data is not a list (i.e., an actual number)
                    return data / scalar  # Perform true division

            # Apply scalar division to the entire nested list structure
            new_data = scalar_div(self.__veri, other)
            return gergen(new_data)
        elif isinstance(other, gergen):
            # Element-wise division of two gergen objects
            if self.__boyut != other.__boyut:
                raise ValueError("Cannot divide gergens with different dimensions.")
            else:
                # Recursive function for element-wise division
                def elementwise_div(data1, data2):
                    if isinstance(data1, list) and isinstance(data2, list):
                        # If both items are lists, recur for each item
                        return [elementwise_div(subdata1, subdata2) for subdata1, subdata2 in zip(data1, data2)]
                    else:
                        # Base case: data1 and data2 are not lists (i.e., actual numbers)
                        return data1 / data2  # Perform true division
                    
                # Apply element-wise division to the entire nested list structure
                new_data = elementwise_div(self.__veri, other.__veri)
                return gergen(new_data)
        else:
            raise TypeError("Divisor must be a scalar or a gergen object.")    
        
    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the addition operation for gergen objects.
        Called when a gergen object is added to another, using the '+' operator.
        The operation is element-wise.
        """
        if self.__veri is None:
            raise ValueError("Cannot add to an empty gergen.")
        
        if isinstance(other, (int, float)):
            if isinstance(self.__veri, (int, float)):
                # Add two scalars
                new_data = self.__veri + other
                return gergen(new_data)
            else:
                # Add the scalar to each element of the gergen of any size
                def scalar_add(data, scalar):
                    if isinstance(data, list):
                        return [scalar_add(subdata, scalar) for subdata in data]
                    else:
                        return data + scalar
                new_data = scalar_add(self.__veri, other)
                return gergen(new_data)
        elif isinstance(other, gergen):
            # Element-wise addition of two gergen objects
            if self.__boyut != other.__boyut:
                raise ValueError("Cannot add gergens with different dimensions.")
            else:
                # Add each element of the gergen of any size to the corresponding element of the other gergen
                def elementwise_add(data1, data2):
                    if isinstance(data1, list) and isinstance(data2, list):
                        return [elementwise_add(subdata1, subdata2) for subdata1, subdata2 in zip(data1, data2)]
                    else:
                        return data1 + data2
                new_data = elementwise_add(self.__veri, other.__veri)
                return gergen(new_data)
        else:
            raise TypeError("Must be a scalar or a gergen object.")

    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Subtraction operation for gergen objects.
        Called when a gergen object is subtracted from another, using the '-' operator.
        The operation is element-wise.
        """
        if self.__veri is None:
            raise ValueError("Cannot subtract from an empty gergen.")
        
        if isinstance(other, (int, float)):
            if isinstance(self.__veri, (int, float)):
                # Subtract two scalars
                new_data = self.__veri - other
                return gergen(new_data)
            else:
                # Subtract the scalar from each element of the gergen of any size
                def scalar_sub(data, scalar):
                    if isinstance(data, list):
                        return [scalar_sub(subdata, scalar) for subdata in data]
                    else:
                        return data - scalar
                new_data = scalar_sub(self.__veri, other)
                return gergen(new_data)
        elif isinstance(other, gergen):
            # Element-wise subtraction of two gergen objects
            if self.__boyut != other.__boyut:
                raise ValueError("Cannot subtract gergens with different dimensions.")
            else:
                # Subtract each element of the gergen of any size from the corresponding element of the other gergen
                def elementwise_sub(data1, data2):
                    if isinstance(data1, list) and isinstance(data2, list):
                        return [elementwise_sub(subdata1, subdata2) for subdata1, subdata2 in zip(data1, data2)]
                    else:
                        return data1 - data2
                new_data = elementwise_sub(self.__veri, other.__veri)
                return gergen(new_data)
        else:
            raise TypeError("Must be a scalar or a gergen object.")

    def uzunluk(self):
        """
        Returns the total number of elements in the gergen.
        """
        if self.__veri is None:
            raise ValueError("Cannot get the length of an empty gergen.")
        else:
            size = 1
            for boyut in self.__boyut: # scalar = 1 or 0 ???
                size *= boyut
            return size

    def boyut(self):
        """
        Returns the shape of the gergen.
        """
        if self.__veri is None:
            raise ValueError("Cannot get the shape of an empty gergen.")
        else:
            return self.__boyut
        
    def devrik(self):
        """
        Returns the transpose of the gergen.
        """
        if self.__veri is None:
            raise ValueError("Cannot get the transpose of an empty gergen.")
        
        if isinstance(self.__veri, (int, float)) or self.boyut() == (1,1):
            return self
        
        if self.boyut()[0] == 1 and len(self.boyut()) == 2:
            return gergen([[item] for item in self.__veri])
        elif self.boyut()[1] == 1 and len(self.boyut()) == 2:
            return gergen([item[0] for item in self.__veri])

        ### for each element in the indices i j k l should be now in l k j i
        # write all possible indices for self.__veri
        def indices(dimensions, idx=[]):
            """
            Generates all possible indices for given dimensions.
            """
            if not dimensions:
                yield idx
                return
            for i in range(dimensions[0]):
                yield from indices(dimensions[1:], idx + [i,])

        # apply the indices to the new_data
        old_indices = list(indices(self.boyut()))

        # find new indices by reversing all of the elements on old_indices
        new_indices = [idx[::-1] for idx in old_indices]
        
        # Initialize new_data with the same structure as self.__veri but empty
        new_data = self.boyutlandir(self.boyut()[::-1]).listeye()
        
        # Function to get/set values in a nested list using a tuple index
        def getset_nested(data, idx, value=None, set_value=False):
            for i in idx[:-1]:
                data = data[i]
            if set_value:
                data[idx[-1]] = value
            else:
                return data[idx[-1]]
        
        # Reassign elements from old to new positions
        for old_idx, new_idx in zip(old_indices, new_indices):
            value = getset_nested(self.__veri, tuple(old_idx))
            getset_nested(new_data, new_idx, value, True)
        
        return gergen(new_data)

    def __apply_elementwise(self, func):
        """
        Applies the given function 'func' element-wise to the gergen.
        """
        if self.__veri is None:
            raise ValueError("Cannot apply the function to an empty gergen.")
        
        def apply_recursive(data):
            if isinstance(data, list):
                return [apply_recursive(subdata) for subdata in data]
            else:
                if func == math.tan and data == math.pi / 2:
                    raise ValueError("Cannot calculate the tangent of pi/2.")
                elif func == math.log10 and data <= 0:
                    raise ValueError("Cannot calculate the logarithm of a non-positive number.")
                elif func == math.log and data <= 0:
                    raise ValueError("Cannot calculate the natural logarithm of a non-positive number.")
                return 0 if abs(func(data)) < 1e-8 else func(data)
            
        new_data = apply_recursive(self.__veri)
        return gergen(new_data)

    def sin(self):
        """
        Calculates the sine of each element in the given gergen.
        """
        return self.__apply_elementwise(math.sin)

    def cos(self):
        """
        Calculates the cosine of each element in the given gergen.
        """
        return self.__apply_elementwise(math.cos)

    def tan(self):
        """
        Calculates the tangent of each element in the given gergen.
        """
        return self.__apply_elementwise(math.tan)

    def us(self, n: int):
        """
        Raises each element of the gergen object to the power 'n'. This is an element-wise operation.
        """
        if self.__veri is None:
            raise ValueError("Cannot raise an empty gergen to a power.")
        
        if n < 0:
            raise ValueError("Cannot raise a gergen to a negative power.")
        
        def apply_recursive(data):
            if isinstance(data, list):
                return [apply_recursive(subdata) for subdata in data]
            else:
                if data == 0 and n == 0:
                    raise ValueError("Cannot raise 0 to the power 0.")
                return data ** n
            
        new_data = apply_recursive(self.__veri)
        return gergen(new_data)
        
    def log(self):
        """
        Applies the logarithm function to each element of the gergen object, using the base 10.
        """
        return self.__apply_elementwise(math.log10)

    def ln(self):
        """
        Applies the natural logarithm function to each element of the gergen object.
        """
        return self.__apply_elementwise(math.log)

    def L1(self):
        """
        Calculates the L1 norm. The L1 norm, also known as the
        Manhattan norm, is the sum of the absolute values of
        the elements in the tensor.
        """
        return self.Lp(1) # L1 norm is a special case of Lp norm

    def L2(self):
        """
        Calculates the L2 norm or the Euclidean norm, which is the
        square root of the sum of the squares of the tensor’s elements.
        """
        return self.Lp(2) # L2 norm is a special case of Lp norm

    def Lp(self, p):
        """
        Calculates Lp norm, which is general version of L1 and L2 
        norms by calculating each element to the power of p,
        summing these values, and then taking the p-th root of the result.
        """
        if self.__veri is None:
            raise ValueError("Cannot calculate the Lp norm of an empty gergen.")
        
        if p <= 0:
            raise ValueError("p must be a positive number.")
        
        if isinstance(self.__veri, (int, float)):
            return math.pow((self.__veri ** p), 1 / p)
        elif self.boyut() == (1,1):
            return math.pow((self.__veri[0] ** p), 1 / p)
        
        # Recursive function to calculate the Lp norm
        def calculate_lp(data):
            if isinstance(data, list):
                return math.pow(sum(calculate_lp(subdata) ** p for subdata in data), 1 / p)
            else:
                return data
            
        return calculate_lp(self.__veri)

    def listeye(self):
        """
        Converts the gergen object into a list or a nested list, depending on its dimensions.
        """
        return self.__veri

    def duzlestir(self):
        """
        Converts the gergen object's multi-dimensional structure into a 1D structure, 
        effectively 'flattening' the object.
        """
        if self.__veri is None:
            raise ValueError("Cannot flatten an empty gergen.")
        
        if isinstance(self.__veri, (int, float)): # ??????
            return gergen(self.__veri)
        
        # Recursive function to flatten the gergen
        def flatten(data):
            if isinstance(data, list):
                return [item for sublist in data for item in flatten(sublist)]
            else:
                return [data]
            
        new_data = flatten(self.__veri)
        return gergen(new_data)
        # Change the current gergen to a 1D gergen and return it

        
        
    def boyutlandir(self, yeni_boyut):
        """
        Reshapes the gergen object to a new shape 'yeni_boyut', which is specified as a tuple.
        """
        if self.__veri is None:
            raise ValueError("Cannot reshape an empty gergen.")
        
        yeni_uzunluk = 1
        for boyut in yeni_boyut:
            yeni_uzunluk *= boyut

        if yeni_uzunluk != self.uzunluk():
            raise ValueError("Total size of new gergen must be unchanged.")
        
        if isinstance(self.__veri, (int, float)):
            return gergen([self.__veri])

        duz = self.duzlestir().listeye()

        def build_structure(data, dims):
            if len(dims) == 0:
                return data.pop(0)
            return [build_structure(data, dims[1:]) for _ in range(dims[0])]
        
        new_data = build_structure(duz, list(yeni_boyut))
        return gergen(new_data)   

    def ic_carpim(self, other):
        """
        Calculates the inner (dot) product of this gergen object with another.
        """
        pass

    def dis_carpim(self, other):
        """
        Calculates the outer product of this gergen object with another.
        """
        pass
    
    def topla(self, eksen=None):
        """
        Sums up values in the gergen. If eksen is None, all elements are added.
        If eksen is not None, performs the summation along the specified axis.
        """
        if self.__veri is None:
            raise ValueError("Cannot sum up elements of an empty gergen.")

        # Check if eksen is valid
        if eksen not in (None, 0, 1):
            raise TypeError("eksen must be None, 0, or 1.")

        # Global summation
        if eksen is None:
            return sum(self.duzlestir().__veri)

        # Axis-specific summation
        def sum_axis(data, axis): # 0 means (3x3x2x3)
            if axis == 0:
                pass
                
        summed_data = sum_axis(self.__veri, eksen)

        # For axis-specific summation, the result should be wrapped in a gergen
        return gergen(summed_data)
        
    def ortalama(self, eksen=None):
        """
        Computes the average of elements in the tensor, with the option to compute
        this average across different axes of the tensor based on the eksen parameter.
        """
        if self.__veri is None:
            raise ValueError("Cannot calculate the average of an empty gergen.")

        # Check if eksen is valid
        if eksen not in (None, 0, 1):
            raise TypeError("eksen must be an integer or None.")

        # Global average
        if eksen is None:
            total_sum = self.topla(eksen=None)
            num_elements = self.uzunluk()
            return total_sum / num_elements if num_elements > 0 else 0

        # Verify dimensions for axis-specific average
        if eksen < 0 or eksen >= len(self.__boyut):
            raise ValueError("Specified eksen is out of bounds.")

        # Axis-specific average
        def avg_axis(data, axis, depth=0):
            if depth == axis:
                # We are at the axis to average over
                if isinstance(data[0], list):
                    # Average each sublist recursively and return as gergen
                    averaged = [sum(item) / len(item) for item in zip(*data)]
                    return averaged
                else:
                    # Base case: directly average the elements at this level
                    return sum(data) / len(data) if data else 0
            else:
                # Not yet at the axis, recurse deeper
                return [avg_axis(item, axis, depth + 1) for item in data]

        averaged_data = avg_axis(self.__veri, eksen)

        # For axis-specific average, the result should be wrapped in a gergen
        return gergen(averaged_data)

def main():
    # a main function to test the functions

    cekirdek(2)
    # g = rastgele_dogal((3,1,2,3))
    # g2 = rastgele_gercek((3, 3, 2, 4))
    # print(g)

    # g = gergen([1, 2, 3])
    # print(g.duzlestir().boyut())
    # print()
    # print(g.listeye())
    # print()
    # print(g)

    ### test the random number generators

    # g1 = rastgele_dogal((3, 3))
    # g1 = rastgele_gercek((3, 3, 3))
    # print(g1)
    # print(gergen())
    # print(g1)
    # print(g1[0][-1])

    ### scalar gergens and init

    # g3 = gergen(2)
    # print(g3)
    # print(g3.D)

    # gi = gergen(4)
    # print(gi)
    # print()
    # gii = gergen([4])
    # print(gii)

    # gv = gergen([1, 2, 3])
    # print(gv)
    # print()
    # gh = gergen([[1], [2], [3]])
    # print(gh)

    ### test the __getitem__ method

    # print(g)
    # print()
    # print(g[0])
    # print()
    # print(g[0, 1]) # = print(g[0][1])
    # print()
    # print(g[0, 2, 1])

    ### test the __str__ method

    # print(gergen())
    # print(gergen(2))
    # print(gergen([[1, 2, 3]]))
    # print(gergen([[1, 2, 3], [4, 5, 6]]))

    # print(g)
    # print()
    # print(g[0])
    # print()
    # print(g[0, 1])
    # print()
    # print(g[0, 2, 1])
    # print()
    # print(g[0, 2, -1, 3])

    ### test the __mul__ method

    # g1 = gergen([[1, 2, 3], [4, 5, 6]])
    # gs1 = g1 * 2
    # # gss1 = g1 * gergen(2) # is not working
    # g2 = gergen([[7, 8, 9], [10, 11, 12]])
    # gs2 = g2 * -3 # (-3 * g2) is not working
    # # gss2 = gergen(-3) * g2 # is not working
    # g3 = g1 * g2
    # g4 = g1 * g2 * 2 # 
    # g5 = g1 * g2 * g3 * 2 # 
    # g1d1 = gergen(4)
    # g1d2 = gergen(3)
    # g1dm = g1d1 * g1d2

    # print(g3) # [[7, 16, 27], [40, 55, 72]]
    # print()
    # print(gs1) 
    # print()
    # print(gs2)
    # print()
    # print(g4) # [[14, 32, 54], [80, 110, 144]]
    # print()
    # print(g5) 
    # print()
    # print(g1dm)
    # print()
    # print(g1dm * g3)

    ### test the __truediv__ method

    # g1 = gergen([[1, 2, 3], [4, 5, 6]])
    # z1 = 0
    # gz1 = g1 / z1
    # print(gz1)

    # g2 = g1 * 2
    # z2 = 2
    # gz2 = g2 / z2
    # print(g2)
    # print()
    # print(gz2)

    # g3 = g1 / g1
    # print(g3)

    # g4 = gergen([[7, 8, 9], [10, 11, 12]])
    # print(g4 * 4 / g4)

    # g5 = gergen([6])
    # print(g5 / 3)
    # g6 = gergen([6, 7])
    # # print(g5 / g6) # ValueError: Cannot divide gergens with different dimensions.
    # g7 = gergen([3])
    # print(g5 / g7)

    ### test the __add__ method

    ## add two scalars
    # gs1 = gergen(2)
    # gs2 = gs1 + 3
    # print(gs2)
    # gs3 = gs1 + gs2
    # print(gs3)

    ## add a scalar to a gergen
    # g = gergen([[1, 2, 3], [4, 5, 6]])
    # gs = g + 2
    # print(gs)

    ## add two gergens
    # ga = gergen([[1, 2, 3], [4, 5, 6]])
    # gb = gergen([[7, 8, 9], [10, 11, 12]])
    # gc = ga + gb
    # print(gc) # [[8, 10, 12], [14, 16, 18]]

    ## add 2 1x1 gergens
    # g11 = gergen([1])
    # g12 = gergen([4])
    # g13 = g11 + g12
    # print(g13)

    ## add 2 1x3 gergens
    # g21 = gergen([1, 2, 3])
    # g22 = gergen([4, 5, 6])
    # g23 = g21 + g22
    # print(g23) # [[5, 7, 9]]

    ## add 2 2x1 gergens
    # g211 = gergen([[1], [2]])
    # g212 = gergen([[3], [4]])
    # g213 = g211 + g212
    # print(g213) # [[4], [6]]

    ## add 1x1 and 1x3 gergens - ValueError: Cannot add gergens with different dimensions.
    # g11 = gergen([1])
    # g13 = gergen([4, 5, 6])
    # g113 = g11 + g13
    # print(g113)

    ## add 1x2 and 2x1 gergens - ValueError: Cannot add gergens with different dimensions.
    # g21 = gergen([1, 2])
    # g212 = gergen([[3], [4]])
    # g213 = g21 + g212
    # print(g213)

    ## add scalar to 5x3x1x2 gergen
    # g53 = rastgele_dogal((5, 3, 1, 2))
    # print(g53)
    # print()
    # gs53 = g53 + 2
    # print(gs53)

    ## add 3x3x2x1 and 3x3x2x1 gergens
    # gg1 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg2 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg3 = gg1 + gg2
    # print(gg3)      

    ### test the __sub__ method

    ## subtract two scalars
    # gs1 = gergen(2)
    # gs2 = gs1 - 3
    # print(gs2)
    # gs3 = gs1 - gs2
    # print(gs3)

    ## subtract a scalar from a gergen
    # g = gergen([[1, 2, 3], [4, 5, 6]])
    # gs = g - 2
    # print(gs)

    ## subtract two gergens
    # ga = gergen([[1, 2, 3], [4, 5, 6]])
    # gb = gergen([[7, 8, 9], [10, 11, 12]])
    # gc = ga - gb
    # print(gc) # [[-6, -6, -6], [-6, -6, -6]]

    ## subtract 2 1x1 gergens
    # g11 = gergen([1])
    # g12 = gergen([4])
    # g13 = g11 - g12
    # print(g13)

    ## subtract 2 1x3 gergens
    # g21 = gergen([1, 2, 3])
    # g22 = gergen([4, 5, 6])
    # g23 = g21 - g22
    # print(g23) # [[-3, -3, -3]]

    ## subtract 2 2x1 gergens
    # g211 = gergen([[1], [2]])
    # g212 = gergen([[3], [4]])
    # g213 = g211 - g212
    # print(g213) # [[-2] \n [-2]]

    ## subtract 1x1 and 1x3 gergens - ValueError: Cannot subtract gergens with different dimensions.
    # g11 = gergen([1])
    # g13 = gergen([4, 5, 6])
    # g113 = g11 - g13
    # print(g113)

    ## subtract 1x2 and 2x1 gergens - ValueError: Cannot subtract gergens with different dimensions.
    # g21 = gergen([1, 2])
    # g212 = gergen([[3], [4]])
    # g213 = g21 - g212
    # print(g213)

    ## subtract scalar from 5x3x1x2 gergen
    # g53 = rastgele_dogal((5, 3, 1, 2))
    # print(g53)
    # print()
    # gs53 = g53 - 2
    # print(gs53)

    ## subtract 3x3x2x1 and 3x3x2x1 gergens
    # gg1 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg2 = gergen([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]], [[[13], [14]], [[15], [16]], [[17], [18]]]])
    # gg3 = gg1 - gg2
    # print(gg3)

    ### test the uzunluk method

    ## empty gergen
    # g = gergen()
    # print(g.uzunluk()) # ValueError: Cannot get the length of an empty gergen.

    ## 1x1 gergen
    # g = gergen([1])
    # print(g.uzunluk()) # 1

    # gl = gergen([[1, 2, 3], [4, 5, 6]])
    # print(gl.uzunluk()) # 6

    # gll = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    # print(gll.uzunluk()) # 12

    # glong = rastgele_dogal((5, 3, 1, 2, 4, 9))
    # print(glong.uzunluk()) # 1080

    # garr = gergen([1,2,3,4,5])
    # print(garr.uzunluk()) # 5

    ### test the boyut method

    ## empty gergen
    # g = gergen()
    # print(g.boyut()) # ValueError: Cannot get the shape of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.boyut()) # (1,1)

    ### test the devrik method

    ## empty gergen
    # g = gergen()
    # print(g.devrik()) # ValueError: Cannot get the transpose of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.devrik()) # 10

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g)
    # print()
    # print(g.devrik()) # [[1] \n [2] \n [3]]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g)
    # print()
    # print(g.devrik()) # [[1, 2, 3]]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.devrik()) # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g)
    # print()
    # print(g.boyutlandir((2,3,3)))
    # print()
    # print(g.devrik()) 
    # print("--------------------")
    # arr = np.array([[[1, 2], [3, 4], [5, 6]],
    #                 [[7, 8], [9, 10], [11, 12]],
    #                 [[13, 14], [15, 16], [17, 18]]])
    # print(arr.transpose().shape)
    # print(arr.transpose())

    ## 3x4x2x5 gergen
    # g = rastgele_dogal((3, 4, 2, 5))
    # print(g)
    # print()
    # print(g.devrik())
    # print("--------------------")
    # arr = np.array(g.listeye())
    # print(arr.transpose().shape)
    # print(arr.transpose())

    ### test the sin, cos, tan methods

    ## empty gergen
    # g = gergen()
    # print(g.sin()) # ValueError: Cannot apply the function to an empty gergen.
    # print(g.cos()) # ValueError: Cannot apply the function to an empty gergen.
    # print(g.tan()) # ValueError: Cannot apply the function to an empty gergen.

    ## 1x1 gergen
    # g = gergen([math.pi / 2])
    # print(g.sin()) # 1.0
    # print(g.cos()) # 0.0
    # print(g.tan()) # err

    ## 1x3 gergen
    # g = gergen([math.pi / 6, math.pi / 3, math.pi / 4])
    # print(g.sin()) 
    # print(g.cos()) 
    # print(g.tan()) 

    ## 3x1 gergen
    # g = gergen([[math.pi / 6], [math.pi / 3], [math.pi / 4]])
    # print(g.sin()) # [[0.5], [0.8660254037844386], [0.7071067811865475]]
    # print(g.cos()) # [[0.8660254037844386], [0.5], [0.7071067811865475]]
    # print(g.tan()) # [[0.5773502691896257], [1.7320508075688774], [0.9999999999999999]]

    ## 3x3 gergen
    # g = gergen([[math.pi / 6, math.pi / 3, math.pi / 4], [math.pi / 6, math.pi / 3, math.pi / 4], [math.pi / 6, math.pi / 3, math.pi / 4]])
    # print(g.sin()) 
    # print(g.cos())
    # print(g.tan())

    ## 3x3x2 gergen
    # g = gergen([[[math.pi / 6, math.pi / 3], [math.pi / 4, math.pi / 6], [math.pi / 3, math.pi / 4]], [[math.pi / 6, math.pi / 3], [math.pi / 4, math.pi / 6], [math.pi / 3, math.pi / 4]], [[math.pi / 6, math.pi / 3], [math.pi / 4, math.pi / 6], [math.pi / 3, math.pi / 4]]])
    # print(g.sin())
    # print(g.cos())
    # print(g.tan())

    ### test the us method

    ## empty gergen
    # g = gergen()
    # print(g.us(2)) # ValueError: Cannot raise an empty gergen to a power.

    ## scalar gergen
    # g = gergen(3)
    # print(g.us(2)) # 9

    # g = gergen(0)
    # print(g.us(0)) # err

    ## 1x1 gergen
    # g = gergen([3])
    # print(g.us(2)) # 9

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.us(2)) # [1, 4, 9]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.us(2)) # [[1], [4], [9]]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.us(2)) # [[1, 4, 9], [16, 25, 36], [49, 64, 81]]

    ## 3x3 gergen with -
    # g = gergen([[4, 4, 4], [4, 4, 4], [4, 4, 4]])
    # print(g.us(-1)) # err

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.us(2))

    ## 3x5x3x2 gergen with 0
    # g = gergen([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]], [[25, 26], [27, 28], [29, 30]]], [[[31, 32], [33, 34], [35, 36]], [[37, 38], [39, 40], [41, 42]], [[43, 44], [45, 46], [47, 48]], [[49, 50], [51, 52], [53, 54]], [[55, 56], [57, 58], [59, 60]]], [[[61, 62], [63, 64], [65, 66]], [[67, 68], [69, 70], [71, 72]], [[73, 74], [75, 76], [77, 78]], [[79, 80], [81, 82], [83, 84]], [[85, 86], [87, 88], [89, 90]]]])
    # print(g.us(0))

    ### test the log methods

    ## empty gergen
    # g = gergen()
    # print(g.log()) # ValueError: Cannot calculate the logarithm of an empty gergen.
    # print(g.ln()) # ValueError: Cannot calculate the natural logarithm of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.log()) # 1.0
    # print(g.ln()) # 2.302585092994046

    ## scalar gergen
    # g = gergen(10)
    # print(g.log()) # 1.0
    # print(g.ln()) # 2.302585092994046

    # g = gergen(0)
    # print(g.log()) # err
    # print(g.ln()) # err

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.log()) # [0.0, 0.3010299956639812, 0.47712125471966244]
    # print(g.ln()) # [0.0, 0.6931471805599453, 1.0986122886681098]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.log()) # [[0.0], [0.3010299956639812], [0.47712125471966244]]
    # print(g.ln()) # [[0.0], [0.6931471805599453], [1.0986122886681098]]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.log()) # [[0.0, 0.3010299956639812, 0.47712125471966244], [0.6020599913279624, 0.6989700043360189, 0.7781512503836436], [0.8450980400142568, 0.9030899869919435, 0.9542425094393249]]
    # print(g.ln()) # [[0.0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055], [1.9459101490553132, 2.0794415416798357, 2.1972245773362196]]

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.log())
    # print(g.ln())

    ### test the L1, L2, Lp methods

    ## empty gergen
    # g = gergen()
    # print(g.L1()) # ValueError: Cannot calculate the L1 norm of an empty gergen.
    # print(g.L2()) # ValueError: Cannot calculate the L2 norm of an empty gergen.
    # print(g.Lp(3)) # ValueError: Cannot calculate the Lp norm of an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.L1()) # 10
    # print(g.L2()) # 10.0
    # print()
    # print(g.Lp(1)) # 10
    # print(g.Lp(2)) # 10.0
    # print(g.Lp(3)) # 10.0

    ## scalar gergen
    # g = gergen(10)
    # print(g.L1()) # 10
    # print(g.L2()) # 10.0
    # print()
    # print(g.Lp(1)) # 10
    # print(g.Lp(2)) # 10.0
    # print(g.Lp(3)) # 10.0

    # print(g.Lp(0)) # ValueError: p must be a positive number.
    # print(g.Lp(-1)) # ValueError: p must be a positive number.

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.L1()) # 6
    # print(g.L2()) # 3.7416573867739413
    # print()
    # print(g.Lp(1)) # 6
    # print(g.Lp(2)) # 3.7416573867739413
    # print(g.Lp(3)) # 3.3019272488946263

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.L1()) # 6
    # print(g.L2()) # 3.7416573867739413
    # print()
    # print(g.Lp(1)) # 6
    # print(g.Lp(2)) # 3.7416573867739413
    # print(g.Lp(3)) # 3.3019272488946263

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.L1()) # 45
    # print(g.L2()) # 16.881943016134134
    # print()
    # print(g.Lp(1)) # 45
    # print(g.Lp(2)) # 16.881943016134134
    # print(g.Lp(3)) # 

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.L1())
    # print(g.L2())
    # print()
    # print(g.Lp(1))
    # print(g.Lp(2))
    # print(g.Lp(3))

    ### test the duzlestir method

    ## empty gergen
    # g = gergen()    
    # print(g.duzlestir()) # ValueError: Cannot flatten an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.duzlestir()) # [10]

    ## scalar gergen
    # g = gergen(10)
    # print(g.duzlestir()) # [10]

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.duzlestir()) # [1, 2, 3]

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.duzlestir()) # [1, 2, 3]

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.duzlestir()) # [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ## 3x3x2 gergen
    # g = gergen([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]])
    # print(g.duzlestir()) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    ## 3x5x3x2 gergen
    # g = rastgele_dogal((3, 5, 3, 2))
    # print(g.duzlestir())

    ## some 3d gergen
    # g = gergen([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    # print(g)
    # print()
    # print(g.duzlestir()) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    ### test the boyutlandir method

    ## empty gergen
    # g = gergen()
    # print(g.boyutlandir((1, 1))) # ValueError: Cannot reshape an empty gergen.

    ## 1x1 gergen
    # g = gergen([10])
    # print(g.boyutlandir((1, 1))) # [[10]]
    # print(g.boyutlandir((1, 2))) # ValueError: Cannot reshape a gergen to a different size.

    ## scalar gergen
    # g = gergen(10)
    # print(g.boyutlandir((1, 1)))
    # print(g.boyutlandir((1, 1, 5))) # ValueError: Cannot reshape a gergen to a different size.

    ## 1x3 gergen
    # g = gergen([1, 2, 3])
    # print(g.boyutlandir((3, 1))) # [[1] \n [2] \n [3]]
    # print(g.boyutlandir((1, 3))) # [[1, 2, 3]]
    # print(g.boyutlandir((3, 1, 1, 1, 6))) # ValueError: Cannot reshape a gergen to a different size.

    ## 3x1 gergen
    # g = gergen([[1], [2], [3]])
    # print(g.boyutlandir((1, 3))) # [[1, 2, 3]]
    # print(g.boyutlandir((3, 1))) # [[1] \n [2] \n [3]]
    # print(g.boyutlandir((1, 1, 3, 1, 1))) 

    ## 3x3 gergen
    # g = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(g.boyutlandir((1, 9))) # [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    # print(g.boyutlandir((9, 1))) # [[1] \n [2] \n [3] \n [4] \n [5] \n [6] \n [7] \n [8] \n [9]]
    # print(g.boyutlandir((3, 3))) # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # print(g.boyutlandir((3, 1, 1, 3, 1)))
    # print(g.boyutlandir((1, 3, 1, 1, 3, 1, 1, 2))) # ValueError: Cannot reshape a gergen to a different size.

    ## 2x3x3 gergen to 3x2x3
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((3, 2, 3)))

    ## 2x3x3 gergen to 9x2 gergen
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((9, 2)))

    ## 2x3x3 gergen to 1x18 gergen
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((1, 18)))

    ## 2x3x3 gergen to 18x1 gergen
    # g = rastgele_dogal((2, 3, 3))
    # print(g)
    # print()
    # print(g.boyutlandir((18, 1)))

    ### test the ic_carpim method

    ### test the dis_carpim method

    ### test the topla method

    ## empty gergen
    # g = gergen()
    # print(g.topla()) # ValueError: Cannot sum up elements of an empty gergen.

    ## 1x1 gergen
    # g = gergen([3,1,2])
    # print(g.topla(0)) # 3

    ### test the ortalama method
    



if __name__ == "__main__":
    main()