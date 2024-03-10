import random
import math
import numpy as np
import time
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
    
class gergen:
    __veri = None # A nested list of numbers representing the data
    _D = None # Transpose of data
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
        self._D = None

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
                y_boyut = (len(temp),)
                return y_boyut
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

        if len(self.__boyut) == 1:
            if len(index) > 1:
                raise IndexError("Too many indices for the gergen.")
            if index[0] < 0:
                index = (index[0] + self.__boyut[0],)
            if index[0] < 0 or index[0] >= self.__boyut[0]:
                raise IndexError("Index out of range")
            return self.__veri[index[0]]
        
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
        elif len(self.__boyut) == 1:
            return f'1 boyutlu vektor gergen:\n{self.__veri}'
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
        
        if isinstance(other, (int, float)) or other.boyut() == (1,):
            if isinstance(self.__veri, (int, float)):
                # Multiply two scalars
                new_data = self.__veri * other if isinstance(other, (int, float)) else self.__veri * other.__veri[0]
                return gergen(new_data)
            else:
                # Multiply each element of the gergen of any size by the scalar
                def scalar_mult(data, scalar):
                    if isinstance(data, list):
                        return [scalar_mult(subdata, scalar) for subdata in data]
                    else:
                        return data * scalar
                new_data = scalar_mult(self.__veri, other) if isinstance(other, (int, float)) else scalar_mult(self.__veri, other.__veri[0])
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
        else:
            raise TypeError("Multiplier must be a scalar or a gergen object.")
            
    def __rmul__(self, other: Union[int, float]) -> 'gergen':
        """
        Handles right-side multiplication, making multiplication commutative.
        """
        # Directly call __mul__ for commutative scalar multiplication
        # This ensures that scalar * gergen uses the same logic as gergen * scalar
        return self.__mul__(other)

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
        if isinstance(other, (int, float)) or other.boyut() == (1,):
            # Recursive function for scalar division
            def scalar_div(data, scalar):
                if isinstance(data, list):
                    # If the current item is a list, recur for each item
                    return [scalar_div(subdata, scalar) for subdata in data]
                else:
                    # Base case: data is not a list (i.e., an actual number)
                    return data / scalar if isinstance(scalar, (int, float)) else data / scalar.__veri[0]

            # Apply scalar division to the entire nested list structure
            new_data = scalar_div(self.__veri, other) if isinstance(other, (int, float)) else scalar_div(self.__veri, other.__veri[0])
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
        
    def __rtruediv__(self, other: Union[int, float]) -> 'gergen':
        """
        Handles right-side division, making division non-commutative.
        """
        # Directly call __truediv__ for non-commutative scalar division
        # This ensures that scalar / gergen makes scalar over each element of gergen
        return self.__reciprocal().__mul__(other)
    
    def __reciprocal(self) -> 'gergen':
        """
        Returns the reciprocal of the gergen object.
        """
        if self.__veri is None:
            raise ValueError("Cannot calculate the reciprocal of an empty gergen.")
        
        # Recursive function to calculate the reciprocal
        def reciprocal_recursive(data):
            if isinstance(data, list):
                return [reciprocal_recursive(subdata) for subdata in data]
            else:
                if data == 0:
                    raise ZeroDivisionError("Cannot calculate the reciprocal of zero.")
                return 1 / data
            
        new_data = reciprocal_recursive(self.__veri)
        return gergen(new_data)
        
    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the addition operation for gergen objects.
        Called when a gergen object is added to another, using the '+' operator.
        The operation is element-wise.
        """
        if self.__veri is None:
            raise ValueError("Cannot add to an empty gergen.")
        
        if isinstance(other, (int, float)) or other.boyut() == (1,):
            if isinstance(self.__veri, (int, float)):
                # Add two scalars
                new_data = self.__veri + other if isinstance(other, (int, float)) else self.__veri + other.__veri[0]
                return gergen(new_data)
            else:
                # Add the scalar to each element of the gergen of any size
                def scalar_add(data, scalar):
                    if isinstance(data, list):
                        return [scalar_add(subdata, scalar) for subdata in data]
                    else:
                        return data + scalar 
                new_data = scalar_add(self.__veri, other) if isinstance(other, (int, float)) else scalar_add(self.__veri, other.__veri[0])
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
        
    def __radd__(self, other: Union[int, float]) -> 'gergen':
        """
        Handles right-side addition, making addition commutative.
        """
        # Directly call __add__ for commutative scalar addition
        # This ensures that scalar + gergen uses the same logic as gergen + scalar
        return self.__add__(other)

    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Subtraction operation for gergen objects.
        Called when a gergen object is subtracted from another, using the '-' operator.
        The operation is element-wise.
        """
        if self.__veri is None:
            raise ValueError("Cannot subtract from an empty gergen.")
        
        if isinstance(other, (int, float)) or other.boyut() == (1,):
            if isinstance(self.__veri, (int, float)):
                # Subtract two scalars
                new_data = self.__veri - other if isinstance(other, (int, float)) else self.__veri - other.__veri[0]
                return gergen(new_data)
            else:
                # Subtract the scalar from each element of the gergen of any size
                def scalar_sub(data, scalar):
                    if isinstance(data, list):
                        return [scalar_sub(subdata, scalar) for subdata in data]
                    else:
                        return data - scalar
                new_data = scalar_sub(self.__veri, other) if isinstance(other, (int, float)) else scalar_sub(self.__veri, other.__veri[0])
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
        
    def __rsub__(self, other: Union[int, float]) -> 'gergen':
        """
        Handles right-side subtraction, making subtraction non-commutative.
        """
        # Directly call __sub__ for non-commutative scalar subtraction
        # This ensures that scalar - gergen uses the same logic as gergen - scalar
        self = self.__mul__(-1)
        return self.__add__(other)

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
        
    @property
    def D(self):
        """
        Computes and returns the transpose of the gergen lazily.
        """
        if self._D is None:  # If the transpose hasn't been computed yet
            self._D = self.devrik()  # Compute the transpose and store it in _D
        return self._D
        
    def devrik(self):
        """
        Returns the transpose of the gergen.
        """
        if self.__veri is None:
            raise ValueError("Cannot get the transpose of an empty gergen.")
        
        if isinstance(self.__veri, (int, float)) or self.boyut() == (1,):
            return self
        
        if self.boyut() == (self.uzunluk(),):
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
        if self.__veri is None or other.__veri is None:
            raise ValueError("Cannot calculate the inner product of an empty gergen.")
        
        if isinstance(self.__veri, (int, float)) and isinstance(other.__veri, (int, float)): # Scalar case
            return self.__veri * other.__veri
        
        if len(self.boyut()) == 1 and len(other.boyut()) == 1: # 1D case
            if self.boyut() != other.boyut():
                raise ValueError("Cannot calculate the inner product of 1D gergens with different dimensions.")
            return sum(a * b for a, b in zip(self.__veri, other.__veri))
        elif len(self.boyut()) == 2 and len(other.boyut()) == 2: # 2D case
            if self.boyut()[1] != other.boyut()[0]:
                raise ValueError("Cannot calculate the inner product of 2D gergens with incompatible dimensions.")
            
            result = [[sum(a * b for a, b in zip(row_a, col_b))
                       for col_b in zip(*other.__veri)] for row_a in self.__veri]
            return gergen(result)
        else:
            raise ValueError("Cannot calculate the inner product of gergens with more than 2 dimensions.")

    def dis_carpim(self, other):
        """
        Calculates the outer product of this gergen object with another.
        """
        if self.__veri is None or other.__veri is None:
            raise ValueError("Cannot calculate the outer product of an empty gergen.")
        
        if not isinstance(other, gergen): # Verifying that the other parameter is a gergen object
            raise TypeError("Both operands must be gergen instances.")
        elif len(self.boyut()) != 1 or len(other.boyut()) != 1: # Ensuring both self and other are 1-D vectors
            raise ValueError("Both operands must be 1-D vectors to compute the outer product.")

        # Calculate the outer product and return
        result = [[self_item * other_item for other_item in other.__veri] for self_item in self.__veri]
        return gergen(result)

    def topla(self, eksen=None):
        """
        Sums up the elements of the tensor, optionally along a specified axis.
        """
        if self.__veri is None:
            raise ValueError("Cannot calculate the sum of an empty gergen.")
        
        def sum_along_the_axis(data, axis):
            if axis == 0:
                if isinstance(data[0], list):
                    return [sum_along_the_axis([x[i] for x in data], 0) for i in range(len(data[0]))]
                else:
                    return sum(data)
            else:
                return [sum_along_the_axis(sublist, axis - 1) for sublist in data]
            
        if eksen is None:
            if isinstance(self.__veri, (int, float)):
                return self.__veri
            return sum(self.duzlestir().__veri)
        else:
            if isinstance(eksen, int) and 0 <= eksen < len(self.__boyut):
                return gergen(sum_along_the_axis(self.__veri, eksen))
            else:
                raise ValueError("Specified axis is out of the bounds")
            
    def ortalama(self, eksen=None):
        """
        Computes the average of elements in the tensor, with the option to compute
        this average across different axes of the tensor based on the eksen parameter.
        """
        if self.__veri is None:
            raise ValueError("Cannot calculate the mean of an empty gergen.")
        
        if eksen is None:
            return self.topla() / self.uzunluk()
        else:
            return self.topla(eksen) / self.__boyut[eksen]

def example_1():
    # Generate two gergen objects A and B with shapes (64, 64) using a similar approach to rastgele_gercek
    # For simplicity, let's assume rastgele_gercek generates a 2D array of random floating-point numbers within a given range
    A_gergen = rastgele_gercek((64, 64))
    A_arr = np.array(A_gergen.listeye())
    B_gergen = rastgele_gercek((64, 64))
    B_arr = np.array(B_gergen.listeye())

    start_time_gergen = time.time()
    ATB_gergen = A_gergen.D * B_gergen # Perform the operation A^T × B (Element-wise multiplication)
    # print(ATB_gergen)
    end_time_gergen = time.time()

    # Now with NumPy
    start_time_numpy = time.time()
    ATB_numpy = A_arr.T * B_arr
    # print(ATB_numpy)
    end_time_numpy = time.time()

    # Time and result comparison
    time_gergen = end_time_gergen - start_time_gergen
    time_numpy = end_time_numpy - start_time_numpy
    difference = [a - b for a, b in zip(ATB_gergen.duzlestir(), ATB_numpy.flatten())] # Differences between elements

    return time_gergen, time_numpy, difference

def example_2():
    # Generate three gergen objects A, B, and C with shapes (4,16,16,16) using a similar approach to rastgele_gercek
    A = rastgele_gercek((4,16,16,16))
    A_arr = np.array(A.listeye())
    B = rastgele_gercek((4,16,16,16))
    B_arr = np.array(B.listeye())
    C = rastgele_gercek((4,16,16,16))
    C_arr = np.array(C.listeye())

    start_time_gergen = time.time()
    result_gergen = (A * B + C * A + B * C).ortalama()
    print("result_gergen: ", result_gergen)
    end_time_gergen = time.time()

    # NumPy equivalent
    start_time_numpy = time.time()
    result_numpy = (A_arr * B_arr + C_arr * A_arr + B_arr * C_arr).mean()
    print("result_numpy: ", result_numpy)
    end_time_numpy = time.time()

    # Time and result comparison
    time_gergen = end_time_gergen - start_time_gergen
    time_numpy = end_time_numpy - start_time_numpy
    difference = np.abs(result_gergen - result_numpy)

    return time_gergen, time_numpy, difference

def example_3():
    # Generate two gergen’s A and B with shapes (3,64,64) using a similar approach to rastgele_gercek
    A = rastgele_gercek((3,64,64))
    A_arr = np.array(A.listeye())
    B = rastgele_gercek((3,64,64))
    B_arr = np.array(B.listeye())

    start_time_gergen = time.time()
    result_gergen = (A.sin() + B.cos()).ln().us(2) / 8  # Perform the operation ln(sin(A) + cos(B))**2 / 8
    # print("result_gergen: ", result_gergen)
    end_time_gergen = time.time()

    # NumPy equivalent
    start_time_numpy = time.time()
    result_numpy = (np.log(np.sin(A_arr) + np.cos(B_arr))**2 / 8)
    # print("result_numpy: ", result_numpy)
    end_time_numpy = time.time()

    # Time and result comparison
    time_gergen = end_time_gergen - start_time_gergen
    time_numpy = end_time_numpy - start_time_numpy
    difference = [a - b for a, b in zip(result_gergen.duzlestir(), result_numpy.flatten())]

    return time_gergen, time_numpy, difference

if __name__ == "__main__":
    # Run and print the results for each example
    time_gergen_1, time_numpy_1, diff_1 = example_1()
    print(f"Example 1: Gergen time: {time_gergen_1:.8f} seconds, NumPy time: {time_numpy_1:.8f} seconds, Maximum difference: {max(diff_1)}")
    time_gergen_2, time_numpy_2, diff_2 = example_2()
    print(f"Example 2: Gergen time: {time_gergen_2:.8f} seconds, NumPy time: {time_numpy_2:.8f} seconds, Difference: {diff_2}")
    time_gergen_3, time_numpy_3, diff_3 = example_3()
    print(f"Example 3: Gergen time: {time_gergen_3:.8f} seconds, NumPy time: {time_numpy_3:.8f} seconds, Maximum difference: {max(diff_3)}")
    