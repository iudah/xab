extern "C"
#ifdef WIN32
    __declspec(dllimport)
#endif
        int main_();

class Model
{
};

class Sequential : public Model
{
};
#if 0
Model define_descriminator(int n_input = 2) {
  Model model = Sequential();
  model.add(Dense(15, n_input, "relu", "he_uniform"));
  model.add(Dense(1, "sigmoid"))
}
#endif
int main(void)
{
  
    {{{
    return main_();
  }}}
}